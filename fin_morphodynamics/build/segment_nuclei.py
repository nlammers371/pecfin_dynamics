"""
Image segmentation via Cellpose library
"""
from tifffile import TiffWriter
import shutil
import logging
import glob2 as glob
import os
import time
from aicsimageio import AICSImage
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
import numpy as np
from cellpose import models
from cellpose.core import use_gpu
# import fractal_tasks_core
from functions.utilities import path_leaf

# logging = logging.getlogging(__name__)
logging.basicConfig(level=logging.NOTSET)
# __OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def segment_FOV(
    column: np.ndarray,
    model=None,
    do_3D: bool = True,
    anisotropy=None,
    diameter: float = 30,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    min_size=None,
    label_dtype=None,
    pretrain_flag=False,
    return_probs=False
):
    """
    Internal function that runs Cellpose segmentation for a single ROI.

    :param column: Three-dimensional numpy array
    :param model: TBD
    :param do_3D: TBD
    :param anisotropy: TBD
    :param diameter: TBD
    :param cellprob_threshold: TBD
    :param flow_threshold: TBD
    :param min_size: TBD
    :param label_dtype: TBD
    """

    # Write some debugging info
    logging.info(
        f"[segment_FOV] START Cellpose |"
        f" column: {type(column)}, {column.shape} |"
        f" do_3D: {do_3D} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" flow threshold: {flow_threshold}"
    )

    # Actual labeling
    t0 = time.perf_counter()
    if not pretrain_flag:
        mask, flows, styles = model.eval(
            column,
            channels=[0, 0],
            do_3D=do_3D,
            net_avg=False,
            augment=False,
            diameter=diameter,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
        )
    else:
        mask, flows, styles = model.eval(
            column,
            channels=[0, 0],
            do_3D=do_3D,
            min_size=min_size,
            diameter=diameter,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
            net_avg=False,
            augment=False
        )
    if not do_3D:
        mask = np.expand_dims(mask, axis=0)
    t1 = time.perf_counter()

    # Write some debugging info
    logging.info(
        f"[segment_FOV] END   Cellpose |"
        f" Elapsed: {t1-t0:.4f} seconds |"
        f" mask shape: {mask.shape},"
        f" mask dtype: {mask.dtype} (before recast to {label_dtype}),"
        f" max(mask): {np.max(mask)} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" anisotropy: {anisotropy} |"
        f" flow threshold: {flow_threshold}"
    )
    if return_probs:
        probs = flows[2]
    else:
        probs = []

    return mask.astype(label_dtype), probs


def cellpose_segmentation(
    *,
    # Fractal arguments
    raw_data_directory: str,
    save_data_directory: str,
    # Task-specific arguments
    seg_channel_label: Optional[str] = None,
    cell_diameter: float = 30,
    cellprob_threshold: float = 0,
    flow_threshold: float = 0.4,
    output_label_name: Optional[str] = None,
    model_type: Literal["nuclei", "cyto", "cyto2"] = "nuclei",
    pretrained_model: Optional[str] = None,
    overwrite: Optional[bool] = False,
    return_probs: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Run cellpose segmentation on the ROIs of a single OME-NGFF image

    Full documentation for all arguments is still TBD, especially because some
    of them are standard arguments for Fractal tasks that should be documented
    in a standard way. Here are some examples of valid arguments::

        input_paths = ["/some/path/*.zarr"]
        component = "some_plate.zarr/B/03/0"
        metadata = {"num_levels": 4, "coarsening_xy": 2}

    :param raw_data_directory: path to directory containing zarr folders for images to segment
    :param seg_channel_label: Identifier of a channel based on its label (e.g.
                          ``DAPI``). If not ``None``, then ``wavelength_id``
                          must be ``None``.
    :param cell_diameter: Initial diameter to be passed to
                            ``CellposeModel.eval`` method (after rescaling from
                            full-resolution to ``level``).
    :param output_label_name: output name for labels
    :param cellprob_threshold: Parameter of ``CellposeModel.eval`` method.
    :param flow_threshold: Parameter of ``CellposeModel.eval`` method.
    :param model_type: Parameter of ``CellposeModel`` class.
    :param pretrained_model: Parameter of ``CellposeModel`` class (takes
                             precedence over ``model_type``).
    """

    # Read useful parameters from metadata
    min_size = (cell_diameter/4)**3

    # Preliminary check
    if seg_channel_label is None:
        raise ValueError(
            f"{seg_channel_label=} argument must be provided"
        )

    # get list of images
    image_list = sorted(glob.glob(raw_data_directory + "*.nd2"))

    if not os.path.isdir(save_data_directory):
        os.makedirs(save_data_directory)

    for im in range(len(image_list)):
        nd2_path = image_list[im]
        im_name = path_leaf(nd2_path)

        # read the image data
        imObject = AICSImage(nd2_path)
        n_wells = len(imObject.scenes)
        n_time_points = imObject.dims["T"][0]

        # extract key image attributes
        channel_names = imObject.channel_names  # list of channels and relevant info

        pixel_res_raw = np.asarray(imObject.physical_pixel_sizes)
        anisotropy = pixel_res_raw[0] / pixel_res_raw[1]

        # Find channel index
        ind_channel = None
        for ch in range(len(channel_names)):
            lbl = channel_names[ch]
            if lbl == seg_channel_label:
                ind_channel = ch

        if ind_channel == None:
            raise Exception(f"ERROR: Specified segmentation channel ({len(seg_channel_label)}) was not found in data")

        for well_index in range(n_wells):

            imObject.set_scene("XYPos:" + str(well_index))

            for t in range(n_time_points):
                # extract image
                data_zyx = np.squeeze(imObject.get_image_data("CZYX", T=t))

                # Select 2D/3D behavior and set some parameters
                do_3D = data_zyx.shape[0] > 1

                # Preliminary checks on Cellpose model
                if pretrained_model is None:
                    if model_type not in ["nuclei", "cyto2", "cyto"]:
                        raise ValueError(f"ERROR model_type={model_type} is not allowed.")
                else:
                    if not os.path.exists(pretrained_model):
                        raise ValueError(f"{pretrained_model=} does not exist.")

                if output_label_name is None:
                    try:
                        channel_label = channel_names[ind_channel]
                        output_label_name = f"label_{channel_label}"
                    except (KeyError, IndexError):
                        output_label_name = f"label_{ind_channel}"

                segment_flag = True
                label_name = im_name.replace('.nd2', f"_well{well_index:03}_t{t:03}_labels")
                label_path = os.path.join(save_data_directory, label_name)
                if os.path.isdir(label_path + '.tif') and overwrite:
                    pass
                elif os.path.isdir(label_path + '.tif'):
                    segment_flag = False

                if segment_flag:

                    logging.info(
                       f"mask will have shape {data_zyx.shape} "
                    )

                    # Initialize cellpose
                    gpu = use_gpu()
                    if pretrained_model:
                        model = models.CellposeModel(
                            gpu=gpu, pretrained_model=pretrained_model
                        )
                    else:
                        model = models.CellposeModel(gpu=gpu, model_type=model_type)

                    # Initialize other things
                    logging.info(f"Start cellpose_segmentation task for {nd2_path}")
                    logging.info(f"do_3D: {do_3D}")
                    logging.info(f"use_gpu: {gpu}")
                    logging.info(f"model_type: {model_type}")
                    logging.info(f"pretrained_model: {pretrained_model}")
                    logging.info(f"anisotropy: {anisotropy}")

                    # Execute illumination correction
                    image_mask, image_probs = segment_FOV(
                        data_zyx, #data_zyx.compute(),
                        model=model,
                        do_3D=do_3D,
                        anisotropy=anisotropy,
                        label_dtype=np.uint32,
                        diameter=cell_diameter,
                        cellprob_threshold=cellprob_threshold,
                        flow_threshold=flow_threshold,
                        min_size=min_size,
                        pretrain_flag=(pretrained_model != None),
                        return_probs=return_probs
                    )

                    # shape0 = data_zyx_raw.shape
                    # print(shape0)
                    # print(image_mask.shape)
                    #image_mask_1 = resize(image_mask, (image_mask.shape[0], shape0[1], shape0[2]), order=0)
                    # if False: #level == 0:
                    #     image_mask_0 = image_mask.copy()
                    # else:
                    #     image_mask_0 = resize(image_mask, shape0, order=0, anti_aliasing=False, preserve_range=True)


                    with TiffWriter(label_path + '.tif', bigtiff=True) as tif:
                        tif.write(image_mask)

                    if return_probs:
                        prob_name = im_name.replace('.nd2', f"_well{well_index:03}_t{t:03}_probs")
                        prob_path = os.path.join(save_data_directory, prob_name)
                        with TiffWriter(prob_path + '.tif', bigtiff=True) as tif:
                            tif.write(image_probs)

                    # im_name = nd2_path.replace('.zarr', '')
                    # with TiffWriter(im_name + 'tif', bigtiff=True) as tif:
                    #     tif.write(data_zyx)
     

                    logging.info(f"End building pyramids, exit")
                else:
                    print(f"WARNING: {nd2_path}labels already exists. Skipping. Set overwrite=True to overwrite")

    return {}

if __name__ == "__main__":
    #raw_data_directory = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"
    raw_data_directory = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\pec_fin_dynamics\\fin_morphodynamics\\raw_data\\20230830\\" #"/mnt/nas/HCR_data/built_zarr_files/"
    save_directory = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\pec_fin_dynamics\\fin_morphodynamics\\built_data\\20230830\\"
    pretrained_model = "C:\\Users\\nlammers\\Projects\\pec_fin_dynamics\\fin_morphodynamics\\cellpose_models\\nuclei_3D_gen_v1"
    overwrite = True
    model_type = "nuclei"
    output_label_name = "td-Tomato"
    seg_channel_label = "561"

    cellpose_segmentation(raw_data_directory=raw_data_directory, save_data_directory=save_directory,
                          seg_channel_label=seg_channel_label, return_probs=True,
                          output_label_name=output_label_name, pretrained_model=pretrained_model, overwrite=overwrite)
