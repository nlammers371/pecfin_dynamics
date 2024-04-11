from aicsimageio import AICSImage
import zarr
import numpy as np
import napari
from skimage.transform import resize
import glob2 as glob
import sys
import threading
import os
# read the image data
import nd2
from tqdm import tqdm

# root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
# experiment_date = "20231013"

# nd2_list = glob.glob(os.path.join())
# full_filename = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/raw_data/20230830/tdTom_54hpf_pecfin_40x.nd2"

def export_nd2_to_zarr(root, full_filename, experiment_date, overwrite_flag):

    out_dir = os.path.join(root, "built_data", "zarr_image_files", experiment_date, "")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # extract key metadata info
    imObject = nd2.ND2File(full_filename)
    im_array_dask = imObject.to_dask()
    nd2_shape = im_array_dask.shape
    n_time_points = nd2_shape[0]
    n_wells = nd2_shape[1]
    dtype = im_array_dask.dtype
    # res_raw = imObject.voxel_size()
    # scale_vec = np.asarray(res_raw)[::-1]
    print("Exporting well time series to zarr...")
    for well_num in tqdm(range(12, n_time_points)):

        # initialize zarr data store
        filename = experiment_date + f"_well{well_num:04}.zarr"
        zarr_file = os.path.join(out_dir, filename)
        well_shape = tuple([nd2_shape[0]]) + nd2_shape[2:]
        # check for existing zarr file
        prev_flag = os.path.isdir(zarr_file)
        # if os.path.isdir(zarr_file) and (not overwrite_flag):
        #     z = zarr.open(zarr_file, mode='r', shape=well_shape, dtype=dtype, chunks=(1,) + nd2_shape[2:])
        #     zarr_shape = z.shape
        #     nz_flag = np.any(z[-1] != 0)
        #     if np.all(zarr_shape == well_shape) & nz_flag:
        #         save_flag = False
        # if save_flag:
        z = zarr.open(zarr_file, mode='a', shape=well_shape, dtype=dtype, chunks=(1,) + nd2_shape[2:])
        if overwrite_flag | (not prev_flag):
            write_indices = np.arange(n_time_points)
        else:
            write_indices = []
            for t in range(n_time_points):
                z_flag = np.all(z[t] == 0)
                if z_flag:
                    write_indices.append(t)
            write_indices = np.asarray(write_indices)

        # data_tzyx = im_array_dask[write_indices, well_num, :, :, :].compute()

        for t, ti in tqdm(enumerate(write_indices)):
            # pass_flag = False
            # iter_i = 0
            # while (pass_flag == False) and (iter_i < 10):
            #     try:
            data_zyx = im_array_dask[ti, well_num, :, :, :].compute()
            z[ti] = data_zyx
            # pass_flag = True
            #     except:
            #         iter_i += 1

    imObject.close()

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    # full_filename = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
    # full_filename = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/raw_data/20240223/wt_tdTom_timelapse_long.nd2"
    experiment_date = "20240223"
    root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
    overwrite_flag = False

    export_nd2_to_zarr(root, full_filename, experiment_date, overwrite_flag)