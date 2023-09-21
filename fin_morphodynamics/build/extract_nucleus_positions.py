# import shutil
# from ome_zarr.reader import Reader
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob2 as glob
from skimage.measure import label, regionprops, regionprops_table
# from scipy.ndimage import distance_transform_edt
from aicsimageio import AICSImage
from alphashape import alphashape
# import math
# import dask
import os
from functions.utilities import path_leaf
import open3d as o3d

# import fractal_tasks_core


def extract_nucleus_stats_prob(data_root, date_folder, voxel_res=2, prob_thresh=0):

    # get paths to raw data and processed probability stacks
    raw_root = os.path.join(data_root, "raw_data", date_folder, "")
    built_root = os.path.join(data_root, "built_data", date_folder, "")
    nucleus_root = os.path.join(data_root, "nucleus_data", date_folder, "")
    if not os.path.isdir(nucleus_root):
        os.makedirs(nucleus_root)

    # get metadata from raw file. If there arte multiple .nd2 files in single date folder, we assume that the pixel
    # sizes are the same for each
    nd2_paths = glob.glob(raw_root + "*.nd2")
    imObject = AICSImage(nd2_paths[0])

    # extract dimension info and generate a scale reference array
    pixel_res_raw = np.asarray(imObject.physical_pixel_sizes)
    im_dims = np.asarray([imObject.dims["Z"][0], imObject.dims["Y"][0], imObject.dims["X"][0]])

    z_vec = np.arange(im_dims[0]) * pixel_res_raw[0]
    y_vec = np.arange(im_dims[1]) * pixel_res_raw[1]
    x_vec = np.arange(im_dims[2]) * pixel_res_raw[2]

    z_ref_array, y_ref_array, x_ref_array = np.meshgrid(z_vec, y_vec, x_vec, indexing="ij")

    # get list of datasets with nucleus labels
    image_list = sorted(glob.glob(built_root + "*_probs.tif"))

    morph_df_list = []
    morph_df_key_list = []

    for im, prob_path in enumerate(tqdm(image_list)):

        # extract metadata
        prob_name = path_leaf(prob_path)
        data_name = prob_name.replace("_prob.tif", "")
        # save_path = os.path.join(nucleus_root, data_name, '_nucleus_props.csv')

        # well number
        well_ind = prob_name.find("well")
        well_num = int(prob_name[well_ind+4:well_ind+7])

        # time step index
        time_ind = int(prob_name[well_ind + 9:well_ind+12])

        if im > 0:
            print("stop")
        print('Extracting stats for ' + data_name)
        probImage = AICSImage(prob_path)
        prob_data = np.squeeze(probImage.data)
        # prob_data_thresh = prob_data > prob_thresh
        # nc_indices = np.where(prob_data_thresh == 1)[0]

        # extract 3D positions of each foreground pixel. This is just a high-res point cloud
        nc_z = z_ref_array[np.where(prob_data > prob_thresh)]
        nc_y = y_ref_array[np.where(prob_data > prob_thresh)]
        nc_x = x_ref_array[np.where(prob_data > prob_thresh)]

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        xyz_array = np.concatenate((nc_x[:, np.newaxis], nc_y[:, np.newaxis], nc_z[:, np.newaxis]), axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_array)
        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_res)
        xyz_ds = np.asarray(pcd_ds.points)

        # convert centroid array to data frame
        morph_df = pd.DataFrame(xyz_ds, columns=["X", "Y", "Z"])

        # add basic metadata
        morph_df["file"] = prob_name
        morph_df["well_id"] = well_num
        morph_df["time_id"] = time_ind

        # generate a dataframe with
        morph_key_df = pd.DataFrame([[prob_name, well_num, time_ind]], columns=["file", "well_id", "time_id"])
        morph_key_df["root_folder"] = data_root
        morph_key_df["experiment_date"] = date_folder
        morph_key_df["voxel_sampling_res_um"] = voxel_res
        morph_key_df["prob_thresh"] = prob_thresh

        # add to list
        morph_df_list.append(morph_df)
        morph_df_key_list.append(morph_key_df)

    # convert to single long dfs
    morph_df_long = pd.concat(morph_df_list, axis=0, ignore_index=True)
    morph_df_key_long = pd.concat(morph_df_key_list, axis=0, ignore_index=True)

    # save
    morph_df_long.to_csv(os.path.join(nucleus_root, "morph_df.csv"))
    morph_df_key_long.to_csv(os.path.join(nucleus_root, "morph_key_df.csv"))

if __name__ == "__main__":
    # define some variables
    root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/"
    date_folder = "20230830"
    prob_thresh = -4
    extract_nucleus_stats_prob(root, date_folder, prob_thresh=prob_thresh, voxel_res=3)
