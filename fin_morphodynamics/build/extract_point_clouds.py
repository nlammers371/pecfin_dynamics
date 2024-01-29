from aicsimageio import AICSImage
import numpy as np
import napari
import os

default_n_threads = 16
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

from glob2 import glob
import skimage.io as io
from alphashape import alphashape
from functions.utilities import path_leaf
from skimage.transform import resize
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans as MiniKMeans, KMeans
# import open3d as o3d


def extract_nucleus_stats_prob(prob_data, prob_thresh=-4.0): #, voxel_res=2):

    # out_shape = np.round(np.multiply(prob_data.shape, pixel_res_vec / pixel_res_vec[0]), 0).astype(int)
    # prob_data_rs = resize(prob_data, out_shape, preserve_range=True, order=1)
    out_shape = prob_data.shape

    z_vec = np.arange(out_shape[0]) #* pixel_res_vec[0]
    y_vec = np.arange(out_shape[1]) #* pixel_res_vec[1]
    x_vec = np.arange(out_shape[2]) #* pixel_res_vec[2]

    z_ref_array, y_ref_array, x_ref_array = np.meshgrid(z_vec, y_vec, x_vec, indexing="ij")

    # extract 3D positions of each foreground pixel. This is just a high-res point cloud
    nc_z = z_ref_array[np.where(prob_data > prob_thresh)]
    nc_y = y_ref_array[np.where(prob_data > prob_thresh)]
    nc_x = x_ref_array[np.where(prob_data > prob_thresh)]

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    zyx_out = np.concatenate((nc_z[:, np.newaxis], nc_y[:, np.newaxis], nc_x[:, np.newaxis]), axis=1)

    return zyx_out

def make_fin_point_clouds(root, experiment_date, prob_thresh, n_points):

    # load metadata
    metadata_path = os.path.join(root, "metadata", experiment_date + "_master_metadata_df.csv")
    metadata_df = pd.read_csv(metadata_path, index_col=0)

    # git list of prob files produced by CellPose
    prob_file_dir = os.path.join(root, "built_data", "cellpose_output", experiment_date, "")
    prob_file_list = glob(prob_file_dir + "*probs*")
    prob_file_list = [file for file in prob_file_list if "full_embryo" not in file]
    prob_file_list = [file for file in prob_file_list if "whole_embryo" not in file]

    pixel_res_vec = metadata_df.loc[0, ["z_res_um", "y_res_um", "x_res_um"]].to_numpy().astype(np.float32)
    # make directory for saving fin masks
    fin_point_dir = os.path.join(root, "built_data", "point_clouds", experiment_date)
    if not os.path.isdir(fin_point_dir):
        os.makedirs(fin_point_dir)
    # determine how many unique embryos and time points we have
    # df_list = []
    for p, prob_name in enumerate(tqdm(prob_file_list)):
        prob_name_short = path_leaf(prob_name)

        # generate save name
        date_ind = prob_name_short.find(experiment_date)
        save_name = prob_name_short[date_ind:].replace(".tif", ".csv")

        if (not os.path.isfile(os.path.join(fin_point_dir, save_name))) | overwrite_flag:
            # well number
            well_ind = prob_name_short.find("well")
            well_num = int(prob_name_short[well_ind + 4:well_ind + 7])

            # time step index
            time_ind = int(prob_name_short[well_ind + 9:well_ind + 12])


            # load the image
            im_prob = io.imread(prob_name, plugin="tifffile")
            # label_name = prob_name.replace("probs", "labels")
            # im_label = io.imread(label_name, plugin="tifffile")
            # obtain point cloud
            zyx_full = extract_nucleus_stats_prob(im_prob, prob_thresh=prob_thresh)
            zyx_full_scaled = np.multiply(zyx_full, pixel_res_vec)
            zyx_full_scaled = zyx_full_scaled.astype(np.float32)

            # use kmeans clustering to downsample the point cloud
            k_clusters = MiniKMeans(n_clusters=n_points, batch_size=10000, n_init='auto').fit(zyx_full_scaled)
            # k_clusters = KMeans(n_clusters=n_points, random_state=0, n_init="auto").fit(zyx_full_scaled)
            ############
            # store results in a pandas dataframe
            point_df = pd.DataFrame(k_clusters.cluster_centers_, columns=["Z", "Y", "X"])
            point_df["prob_thresh"] = prob_thresh
            point_df["experiment_date"] = experiment_date
            point_df["time_id"] = time_ind
            point_df["well_id"] = well_num
            point_df = point_df.iloc[:, ::-1]

            # save
            point_df.to_csv(os.path.join(fin_point_dir, save_name))

        else:
            print(F"Skipping {prob_name_short}...")

    return {}

if __name__ == '__main__':

    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/"
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\"
    experiment_date = "20231214"
    prob_thresh = -8
    n_points = 4096
    overwrite_flag = False

    make_fin_point_clouds(root, experiment_date, prob_thresh, n_points)

