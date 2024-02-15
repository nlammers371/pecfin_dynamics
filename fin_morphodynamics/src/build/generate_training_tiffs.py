import numpy as np
import napari
import os
import glob2 as glob
import cv2
from aicsimageio import AICSImage
import ntpath
from tqdm import tqdm
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from skimage.transform import resize
# from PIL import Image

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


if __name__ == "__main__":

    # designate read paths
    raw_data_directory = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
    date_folder_vec = ["20231013"] #["20231013", "20230913"]
    xy_ds_factor = 1
    edge_buffer_z = 7
    edge_buffer_xy = int(350/xy_ds_factor)
    for d, date_folder in enumerate(date_folder_vec):
        image_directory = os.path.join(raw_data_directory,  "raw_data", date_folder, '')


        suffix_vec = ["_xy", "_zx", "_zy"]
        window_size = 512
        decon_flag = True
        n_samples = 1  # number of samples in xy and zx/zy directions (total num samples per image = 2*n_samples)
        overwrite_flag = False
        skip_labeled_flag = False
        if overwrite_flag:
            skip_labeled_flag = False

        # get list of datasets with label priors available for revision
        # get list of images
        image_list = sorted(glob.glob(image_directory + "*.nd2"))
        seg_channel_label = "561"

        # set write path
        write_path = os.path.join(raw_data_directory, "raw_data", 'cellpose_training_slices', date_folder, '')
        if not os.path.isdir(write_path):
            os.makedirs(write_path)

        # Iterate through images
        for im in range(len(image_list)):

            nd2_path = image_list[im]
            im_name = path_leaf(nd2_path)
            im_name = im_name.replace(".nd2", "")

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
                    image_data_raw = np.squeeze(imObject.get_image_data("CZYX", T=t))
                    save_name = im_name + f"_w{well_index:03}_t{t:03}"

                    dims_orig = image_data_raw.shape
                    dims_new = np.round(
                        [dims_orig[0]*anisotropy, dims_orig[1] / xy_ds_factor, dims_orig[2] / xy_ds_factor]).astype(int)
                    image_data = resize(image_data_raw, dims_new, order=1)

                    # pixel_res_new = pixel_res_raw.copy()
                    # pixel_res_new[1:] = pixel_res_new[1:] * dims_orig[2]/dims_new[2]
                    # pixel_res_new[0] = pixel_res_new[0] *

                    # randomly choose slices along each direction
                    dim_vec = image_data.shape
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

                        slice_path = write_path + save_name + suffix + ".tiff"

                        rand_prefix = np.random.randint(0, 100000, 1)[0]
                        out_name = os.path.join(write_path, f'{rand_prefix:06}' + '_' + date_folder + '_' + save_name + suffix + f'{slice_num:03}')

                        # print("Starting with raw label priors...")
                        if slice_id == 0:
                            im_slice = image_data[slice_num, :, :]

                        elif slice_id == 1:
                            im_slice = np.squeeze(image_data[:, slice_num, :])
                            # rescale
                            # rs_factor = pixel_res_raw[0] / pixel_res_raw[2]
                            # new_dim = int(rs_factor * dim_vec[0])
                            # im_slice = resize(im_slice, [new_dim, dim_vec[2]], order=1, preserve_range=True)

                        else: # slice_id == 2:
                            im_slice = np.squeeze(image_data[:, :, slice_num])
                            # rescale
                            # rs_factor = pixel_res_raw[0] / pixel_res_raw[1]
                            # new_dim = int(rs_factor * dim_vec[0])
                            # im_slice = resize(im_slice, [new_dim, dim_vec[1]], order=1, preserve_range=True)

                        shape_full = np.asarray(im_slice.shape)
                        im_lims = shape_full-window_size
                        x_start = np.random.choice(range(im_lims[1]), 1)[0].astype(int)
                        if im_lims[0] < 0:
                            y_start = 0
                            im_slice_chunk = im_slice[:, x_start:x_start + window_size]
                        else:
                            y_start = np.random.choice(range(im_lims[0]), 1)[0].astype(int)
                            im_slice_chunk = im_slice[y_start:y_start+window_size, x_start:x_start+window_size]

                        if np.max(im_slice_chunk < 1):
                            im_slice_chunk = im_slice_chunk*(2**16)
                        im_slice_out = im_slice_chunk.astype(np.uint16)
                        # write to file
                        cv2.imwrite(out_name + ".tiff", im_slice_out)
