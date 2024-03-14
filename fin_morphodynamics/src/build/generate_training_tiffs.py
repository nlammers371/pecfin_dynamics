import numpy as np
import napari
import os
import glob2 as glob
import skimage.io as io
from aicsimageio import AICSImage
import ntpath
from tqdm import tqdm
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from skimage.transform import resize
import zarr
# from PIL import Image

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def make_training_snips(root, xy_ds_factor=1.0, edge_buffer_z=7, edge_buffer_xy=10, window_size=256, n_samples=2,
                        nuclear_channel=0, overwrite_flag=False, pixel_res_vec=None):


    for d, date_folder in enumerate(date_folder_vec):
        image_directory = os.path.join(root,  "built_data", "zarr_image_files", date_folder, '')

        suffix_vec = ["_xy", "_zx", "_zy"]

        # get list of datasets with label priors available for revision
        # get list of images
        image_list = sorted(glob.glob(image_directory + "*.zarr"))
        # seg_channel_label = "561"

        # set write path
        write_path = os.path.join(root, "raw_data", 'cellpose_training_slices', date_folder, '')
        if not os.path.isdir(write_path):
            os.makedirs(write_path)

        # Iterate through images
        for im in range(len(image_list)):

            zarr_path = image_list[im]
            im_name = path_leaf(zarr_path)
            im_name = im_name.replace(".zarr", "")

            # read the image data
            # imObject = AICSImage(nd2_path)
            # n_wells = len(imObject.scenes)
            # 

            data_tzyx = zarr.open(image_list[im])
            n_time_points = data_tzyx.shape[0]
            
            # extract key image attributes
            # channel_names = imObject.channel_names  # list of channels and relevant info

            # pixel_res_raw = np.asarray(imObject.physical_pixel_sizes)
            anisotropy = pixel_res_vec[0] / pixel_res_vec[1]

            # Find channel index
            # ind_channel = None
            # for ch in range(len(channel_names)):
            #     lbl = channel_names[ch]
            #     if lbl == seg_channel_label:
            #         ind_channel = ch

            # if ind_channel == None:
            #     raise Exception(f"ERROR: Specified segmentation channel ({len(seg_channel_label)}) was not found in data")

            # for well_index in range(n_wells):
            # 
            #     imObject.set_scene("XYPos:" + str(well_index))

            for t in range(n_time_points-2, n_time_points):
                # extract image
                data_zyx_raw = data_tzyx[t]
                save_name = im_name + f"_w{im:03}_t{t:03}"

                dims_orig = data_zyx_raw.shape
                dims_new = np.round(
                    [dims_orig[0]*anisotropy, dims_orig[1] / xy_ds_factor, dims_orig[2] / xy_ds_factor]).astype(int)
                data_zyx = resize(data_zyx_raw, dims_new, order=1)


                # randomly choose slices along each direction
                dim_vec = data_zyx.shape
                xy_slice_indices = np.random.choice(range(edge_buffer_z, dim_vec[0]-edge_buffer_z), n_samples, replace=False)
                xy_id_arr = np.zeros(xy_slice_indices.shape)
                zx_slice_indices = np.random.choice(range(edge_buffer_xy, dim_vec[1]-edge_buffer_xy), int(np.ceil(n_samples / 2)), replace=False)
                zx_id_arr = np.ones(zx_slice_indices.shape)
                zy_slice_indices = np.random.choice(range(edge_buffer_xy, dim_vec[2]-edge_buffer_xy), int(np.ceil(n_samples / 2)), replace=False)
                zy_id_arr = np.ones(zy_slice_indices.shape) * 2

                # combine and shuffle
                slice_num_vec = np.concatenate((xy_slice_indices, zx_slice_indices, zy_slice_indices), axis=0)
                slice_id_vec = np.concatenate((xy_id_arr, zx_id_arr, zy_id_arr), axis=0)
                shuffle_vec = np.random.choice(range(len(slice_id_vec)), len(slice_id_vec), replace=False)
                slice_num_vec = slice_num_vec[shuffle_vec].astype(int)
                slice_id_vec = slice_id_vec[shuffle_vec].astype(int)

                for image_i in range(len(slice_id_vec)):

                    # generate save paths for image slice and labels
                    slice_id = slice_id_vec[image_i]
                    slice_num = slice_num_vec[image_i]
                    suffix = suffix_vec[slice_id]

                    rand_prefix = np.random.randint(0, 100000, 1)[0]
                    out_name = os.path.join(write_path, f'{rand_prefix:06}' + '_' + date_folder + '_' + save_name + suffix + f'{slice_num:03}')

                    # print("Starting with raw label priors...")
                    if slice_id == 0:
                        im_slice = data_zyx[slice_num, :, :]

                    elif slice_id == 1:
                        im_slice = np.squeeze(data_zyx[:, slice_num, :])

                    else:
                        im_slice = np.squeeze(data_zyx[:, :, slice_num])

                    shape_full = np.asarray(im_slice.shape)
                    im_lims = shape_full-window_size
                    x_start = np.random.choice(range(im_lims[1]), 1)[0].astype(int)
                    if im_lims[0] < 0:
                        im_slice_chunk = im_slice[:, x_start:x_start + window_size]
                    else:
                        y_start = np.random.choice(range(im_lims[0]), 1)[0].astype(int)
                        im_slice_chunk = im_slice[y_start:y_start+window_size, x_start:x_start+window_size]

                    if np.max(im_slice_chunk < 1):
                        im_slice_chunk = im_slice_chunk*(2**16)
                    im_slice_out = im_slice_chunk.astype(np.uint16)

                    # write to file
                    io.imsave(out_name + ".tiff", im_slice_out, check_contrast=False)


if __name__ == "__main__":

    # designate read paths
    root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
    date_folder_vec = ["20240223"] #["20231013", "20230913"]
    pixel_res_vec = np.asarray([2.0, 0.55, 0.55])

    make_training_snips(root, pixel_res_vec=pixel_res_vec)

