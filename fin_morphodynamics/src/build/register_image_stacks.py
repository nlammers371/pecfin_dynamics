import napari
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import zoom
import zarr
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
import os
import glob2 as glob
from fin_morphodynamics.src.utilities.functions import path_leaf

def register_timelapse(video: np.ndarray, mask_thresh=None) -> np.ndarray:

    # registration using channel 1
    shift_array = np.zeros((video.shape[0], 3))
    for t in tqdm(range(video.shape[0] - 1), "registering images..."):

        if mask_thresh is None:
            shift, error, _ = phase_cross_correlation(
                video[t].astype(np.float32),
                video[t + 1].astype(np.float32),
                normalization=None,
                upsample_factor=4,
                overlap_ratio=0.25,
            )

        else:
            shift, error, _ = phase_cross_correlation(
                video[t].astype(np.float32),
                video[t + 1].astype(np.float32),
                normalization=None,
                reference_mask=video[t]>=mask_thresh,
                moving_mask=video[t+1]>=mask_thresh,
                upsample_factor=4,
                overlap_ratio=0.25,
            )

        shift_array[t + 1, :] = shift
        # if out_array is not None:
        video[t+1] = ndi.shift(video[t+1], (shift), order=1)

    # if out_array is not None:
    #     return out_array, shift_array
    # else:
    #     return video, shift_array
    return video, shift_array

def registration_wrapper(root, experiment_date, model_name, iso_flag=True, scale_vec=None, overwrite=False):

    if scale_vec is None:
        scale_vec = np.asarray([2.0, 0.55, 0.55])

    # path to zarr files
    data_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')
    reg_data_directory = os.path.join(root, "built_data", "zarr_image_files_registered", experiment_date, '')
    if not os.path.isdir(reg_data_directory):
        os.makedirs(reg_data_directory)

    # get path to cellpose output
    # cellpose_directory = os.path.join(root, "built_data", "cellpose_output", model_name, experiment_date, '')

    # get list of images
    image_list = sorted(glob.glob(data_directory + "*.zarr"))
    
    for well_index in [12]: #range(len(image_list)):

        # file_prefix = path_leaf(well).replace("_probs.zarr", "")
        # prob_name = os.path.join(cellpose_directory, file_prefix + "_probs.zarr")

        # prob_zarr = zarr.open(prob_name, mode="r")
        zarr_path = image_list[well_index]
        im_name = path_leaf(zarr_path)
        print("processing " + im_name)
        # read the image data
        data_zarr = zarr.open(zarr_path, mode="r")

        # generate zarr files
        file_prefix = experiment_date + f"_well{well_index:04}"
        out_zarr_path = os.path.join(reg_data_directory, file_prefix + "_registered.zarr")
        # prev_flag = os.path.isdir(out_zarr_path)
        data_zarr = data_zarr[:30]
        out_zarr = zarr.open(out_zarr_path, mode='a', shape=data_zarr.shape, dtype=np.uitn16,
                             chunks=(1,) + data_zarr.shape[1:])
        out_zarr[:] = data_zarr
        # zarr.copy_all(data_zarr, dest=out_zarr)

        # if iso_flag: # do we want to rescale the z axis to be isotropic
        #     shape_orig = np.asarray(data_zarr[0].shape)
        #     shape_iso = shape_orig.copy()
        #     iso_factor = scale_vec[0] / scale_vec[1]
        #     shape_iso[0] = int(shape_iso[0] * iso_factor)
        #
        #     data_zarr_temp = np.empty((data_zarr.shape[0],) + tuple(shape_iso), dtype=np.uint16)
        #     zoom_factor = np.divide(shape_iso, shape_orig)
        #     for t in tqdm(range(data_zarr.shape[0]), "resizing arrays..."):
        #         data_zarr_temp[t] = zoom(data_zarr[t], zoom_factor, order=1)
            # call registration function

        # data_zarr_chunk = data_zarr[:, 8:26, 400:850, 80:520]

        out_zarr, shift_array = register_timelapse(out_zarr)
        # out_array_temp = np.empty((data_zarr.shape[0],) + tuple(shape_iso), dtype=np.uint16)
        # out_array_temp[0] = data_zarr[0]
        # out_zarr[0] = data_zarr[0]
        # for t in tqdm(range(1, data_zarr.shape[0]), "shifting array..."):
        #     out_zarr[t] = ndi.shift(data_zarr[t], shift_array[t], order=1)
        print("Check")


if __name__ == "__main__":
    overwrite = False

    # set read/write paths
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
    experiment_date = "20240223"
    model_name = "log-v5"

    registration_wrapper(root=root, experiment_date=experiment_date, model_name=model_name, overwrite=overwrite)