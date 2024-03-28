"""
Image segmentation via Cellpose library
"""
from tifffile import TiffWriter
from tqdm import tqdm
import logging
import glob2 as glob
import os
import time
import SimpleITK as sitk
# import pyclesperanto as cle
import skimage as ski
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
import numpy as np
from cellpose import models
from cellpose.core import use_gpu
from skimage.transform import resize
import skimage.io as io
from functions.utilities import path_leaf
import zarr
from fin_morphodynamics.src.utilities.image_utils import calculate_LoG

# logging = logging.getlogging(__name__)
logging.basicConfig(level=logging.NOTSET)
# __OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

def segment_FOV(
    column: np.ndarray,
    model=None,
    do_3D: bool = True,
    anisotropy=None,
    diameter: float = 15,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    min_size=None,
    label_dtype=None,
    pretrain_flag=False,
    return_probs=True,
    return_grads=True
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

    if return_grads:
        grads = flows[1]
    else:
        grads = []

    return mask.astype(label_dtype), probs, grads


def cellpose_segmentation(
    *,
    # Fractal arguments
    root: str,
    experiment_date: str,
    # Task-specific arguments
    # seg_channel_label: Optional[str] = None,
    cell_diameter: float = 30,
    cellprob_threshold: float = 0,
    flow_threshold: float = 0.4,
    output_label_name: Optional[str] = None,
    model_type: Literal["nuclei", "cyto", "cyto2"] = "nuclei",
    pretrained_model: Optional[str] = None,
    overwrite: Optional[bool] = False,
    return_probs: Optional[bool] = True,
    return_grads: Optional[bool] = True,
    xy_ds_factor: Optional[float] = 1.0,
    pixel_res_raw=None,
    file_suffix=".zarr"
    # tiff_stack_mode = False,
    # pixel_res_input = None
) -> Dict[str, Any]:
    """
    Run cellpose segmentation on the ROIs of a single OME-NGFF image

    Full documentation for all arguments is still TBD, especially because some
    of them are standard arguments for Fractal tasks that should be documented
    in a standard way. Here are some examples of valid arguments::

        input_paths = ["/some/path/*.zarr"]
        component = "some_plate.zarr/B/03/0"
        metadata = {"num_levels": 4, "coarsening_xy": 2}

    :param data_directory: path to directory containing zarr folders for images to segment
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
    min_size = 5  # let's be conservative here     # (cell_diameter/4)**3 / xy_ds_factor**2

    # if tiff_stack_mode:
    if pixel_res_raw is None:
        raise Exception("User must input pixel resolutions if using tiff stack mode")

    # path to zarr files
    data_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')

    model_name = path_leaf(pretrained_model)
    save_directory = os.path.join(root, "built_data", "cellpose_output", model_name, experiment_date, '')
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # get list of images
    image_list = sorted(glob.glob(data_directory + "*.zarr"))

    for well_index in range(len(image_list)):

        zarr_path = image_list[well_index]
        im_name = path_leaf(zarr_path)
        print("processing " + im_name)
        # read the image data
        data_tzyx = zarr.open(zarr_path, mode="r")
        # n_wells = len(imObject.scenes)
        # well_list = imObject.scenes
        n_time_points = data_tzyx.shape[0]

        # make sure we are not accidentally up-sampling
        assert xy_ds_factor >= 1.0
        anisotropy_raw = pixel_res_raw[0] / pixel_res_raw[1]

        # generate zarr files
        file_prefix = experiment_date + f"_well{well_index:04}"
        mask_zarr_path = os.path.join(save_directory, file_prefix + "_labels.zarr")
        prev_flag = os.path.isdir(mask_zarr_path)

        mask_zarr = zarr.open(mask_zarr_path, mode='a', shape=data_tzyx.shape, dtype=np.uint16, chunks=(1,) + data_tzyx.shape[2:])
        if return_probs:
            prob_zarr_path = os.path.join(save_directory, file_prefix + "_probs.zarr")
            prob_zarr = zarr.open(mask_zarr_path, mode='a', shape=data_tzyx.shape, dtype=np.uint16,
                      chunks=(1,) + data_tzyx.shape[2:])

        if return_grads:
            grad_zarr_path = os.path.join(save_directory, file_prefix + "_probs.zarr")
            grad_zarr = zarr.open(mask_zarr_path, mode='a', shape=data_tzyx.shape, dtype=np.uint16,
                                  chunks=(1,) + data_tzyx.shape[2:])

        # determine which indices to segment
        if overwrite | (not prev_flag):
            write_indices = np.arange(n_time_points)
        else:
            write_indices = []
            for t in range(n_time_points):
                z_flag = np.all(z[t, :, :, :] == 0)
                if z_flag:
                    write_indices.append(t)
            write_indices = np.asarray(write_indices)

        for t in write_indices:

            # extract image
            data_zyx_raw = data_tzyx[t]

            # rescale data
            dims_orig = data_zyx_raw.shape
            if xy_ds_factor > 1.0:
                dims_new = np.round([dims_orig[0], dims_orig[1]/xy_ds_factor, dims_orig[2]/xy_ds_factor]).astype(int)
                data_zyx = resize(data_zyx_raw, dims_new, order=1)
            else:
                dims_new = dims_orig
                data_zyx = data_zyx_raw.copy()

            if ("log" in model_name) or ("bkg" in model_name):
                im_log, im_bkg = calculate_LoG(data_zyx=data_zyx, scale_vec=pixel_res_raw)
            if "log" in model_name:
                data_zyx = im_log
            elif "bkg" in model_name:
                print("bkg")
                data_zyx = im_bkg

            anisotropy = anisotropy_raw * dims_new[1] / dims_orig[1]

            # Select 2D/3D behavior and set some parameters
            do_3D = data_zyx.shape[0] > 1

            # Preliminary checks on Cellpose model
            if pretrained_model is None:
                if model_type not in ["nuclei", "cyto2", "cyto"]:
                    raise ValueError(f"ERROR model_type={model_type} is not allowed.")
            else:
                if not os.path.exists(pretrained_model):
                    raise ValueError(f"{pretrained_model=} does not exist.")

            # if output_label_name is None:
            #     try:
            #         channel_label = channel_names[ind_channel]
            #         output_label_name = f"label_{channel_label}"
            #     except (KeyError, IndexError):
            #         output_label_name = f"label_{ind_channel}"

            segment_flag = True
            label_name = experiment_date + f"_well{well_index:03}_t{t:03}_labels.tif"
            label_path = os.path.join(save_directory, label_name)
            # if (not os.path.isfile(label_path + '.tif')) | overwrite:
            #     pass
            if os.path.isfile(label_path + '.tif') and (overwrite==False):
                segment_flag = False
                # print("skipping " + label_path)

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
                logging.info(f"Start cellpose_segmentation task for {zarr_path}")
                logging.info(f"do_3D: {do_3D}")
                logging.info(f"use_gpu: {gpu}")
                logging.info(f"model_type: {model_type}")
                logging.info(f"pretrained_model: {pretrained_model}")
                logging.info(f"anisotropy: {anisotropy}")

                # Execute illumination correction
                image_mask, image_probs, im_grads = segment_FOV(
                    data_zyx, #data_zyx.compute(),
                    model=model,
                    do_3D=do_3D,
                    anisotropy=anisotropy,
                    label_dtype=np.uint32,
                    diameter=cell_diameter / xy_ds_factor,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                    min_size=min_size,
                    pretrain_flag=(pretrained_model != None),
                    return_probs=return_probs
                )

                if xy_ds_factor > 1.0:
                    image_mask_out = resize(image_mask, dims_orig, order=0, anti_aliasing=False, preserve_range=True)
                    image_probs_out = resize(image_probs, dims_orig, order=1)

                else:
                    image_mask_out = image_mask
                    image_probs_out = image_probs
                
                # with TiffWriter(label_path + '.tif', bigtiff=True) as tif:
                image_mask_out = image_mask_out.astype(np.uint16)     # save some disk space
                io.imsave(label_path, image_mask_out, check_contrast=False)

                if return_probs:
                    prob_name = experiment_date + f"_well{well_index:03}_t{t:03}_probs.tif"
                    prob_path = os.path.join(save_directory, prob_name)
                    io.imsave(prob_path, image_probs_out, check_contrast=False)

                if return_grads:
                    prob_name = experiment_date + f"_well{well_index:03}_t{t:03}_probs.tif"
                    prob_path = os.path.join(save_directory, prob_name)
                    io.imsave(prob_path, image_probs_out, check_contrast=False)


                # im_name = zarr_path.replace(file_suffix, '')
                # with TiffWriter(im_name + 'tif', bigtiff=True) as tif:
                #     tif.write(data_zyx)
 

                logging.info(f"End file save process, exit")
            else:
                print("skipping " + label_path)

    return {}

if __name__ == "__main__":
    # sert some hyperparameters
    overwrite = False
    model_type = "nuclei"
    output_label_name = "td-Tomato"
    seg_channel_label = "561"
    xy_ds_factor = 1
    cell_diameter = 8
    cellprob_threshold = 0.0
    pixel_res_raw = [2, 0.55, 0.55]
    # set path to CellPose model to use
    pretrained_model = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\built_data\\cellpose_training\\20240223_tdTom\\bk\\models\\bkg-v2"

    # set read/write paths
    root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
    experiment_date = "20240223"

    cellpose_segmentation(root=root, experiment_date=experiment_date, pixel_res_raw=pixel_res_raw,
                          return_probs=True, xy_ds_factor=xy_ds_factor, cell_diameter=cell_diameter,
                          cellprob_threshold=cellprob_threshold,
                          output_label_name=output_label_name, pretrained_model=pretrained_model, overwrite=overwrite)
