# from aicsimageio import AICSImage
import numpy as np
import napari
import os
from src.utilities.functions import path_leaf
from glob2 import glob
import skimage.io as skio
from skimage.measure import regionprops
import pandas as pd
# import open3d as o3d
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
import time
import vispy.color
from tqdm import tqdm
import zarr
from sklearn.neighbors import KDTree
import networkx as nx
from src.utilities.point_cloud_utils import farthest_point_sample

def strip_dummy_cols(df):
    cols = df.columns
    keep_cols = [col for col in cols if "Unnamed" not in col]
    df = df[keep_cols]
    return df


def load_zarr_data(root, seg_model, experiment_date, file_prefix, time_int):

    # path to raw data
    raw_zarr_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, file_prefix + ".zarr")
    mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, experiment_date, file_prefix + "_mask_aff.zarr")
    prob_zarr_path = os.path.join(root, "built_data", "cellpose_output", seg_model, experiment_date,
                                  file_prefix + "_probs.zarr")

    data_zarr = zarr.open(raw_zarr_path, mode="r")
    mask_zarr = zarr.open(mask_zarr_path, mode="r")
    prob_zarr = zarr.open(prob_zarr_path, mode="r")

    # convert scale vec to tuple
    scale_vec = data_zarr.attrs["voxel_size_um"]
    scale_vec = tuple(scale_vec)

    # load the specific time point we want
    time_range = np.arange(np.max([0, time_int-3]), np.min([len(prob_zarr), time_int+4]))
    im_prob = prob_zarr[time_range]
    im_mask = mask_zarr[time_int]

    return im_prob, im_mask, scale_vec


def load_points_and_labels(root, file_prefix, time_int, fluo_flag=False):

    # check for point cloud dataset
    point_prefix = file_prefix + f"_time{time_int:04}"
    point_path = os.path.join(root, "point_cloud_data", "point_features" + "_" + seg_type_global, "")
    point_path_out = os.path.join(root, "point_cloud_data", "fin_segmentation_bin" + "_" + seg_type_global, "")


    if not os.path.isdir(point_path_out):
        os.makedirs(point_path_out)

    point_df_temp = pd.read_csv(point_path + point_prefix + "_points_features.csv")
    point_df_temp = strip_dummy_cols(point_df_temp)
    point_df = point_df_temp.copy()

    # check for pre-existing labels DF
    if os.path.isfile(point_path_out + point_prefix + "_labels.csv"):
        labels_df = pd.read_csv(point_path_out + point_prefix + "_labels.csv")
        labels_df = strip_dummy_cols(labels_df)

    else:
        keep_cols = [col for col in point_df.columns if "feat" not in col]
        labels_df = point_df.loc[:, keep_cols]
        labels_df["fin_curation_flag"] = False
        labels_df["fin_curation_date"] = np.nan
        labels_df["fin_label_curr"] = 0

    if not fluo_flag:
        labels_df.loc[labels_df["fin_label_curr"] == -1, "fin_label_curr"] = 0
    else:
        labels_df["fin_label_curr"] = labels_df["fluo_label"]
    labels_df["fin_label_pd"] = labels_df["fin_label_curr"]

    return point_df, labels_df, point_prefix, point_path_out


def plot_model_features(root, experiment_date, well_num, seg_model, seg_type, feature_num=0, time_int=0):


    # initialize global variables
    global mlp_df, mdl, point_df, labels_df, seg_type_global, train_counter, Y_probs, mask_zarr, label_mask, pd_layer, lb_layer

    seg_type_global = seg_type

    train_counter = 0

    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"

    prob_zarr, mask_zarr, scale_vec = load_zarr_data(root, seg_model, experiment_date, file_prefix, time_int)

    # load point features and labels
    point_df, labels_df, point_prefix, point_path_out = load_points_and_labels(root, file_prefix, time_int, fluo_flag)


    # initialize viewer
    viewer = napari.Viewer()
    viewer.add_image(prob_zarr, colormap="gray", scale=scale_vec,
                               contrast_limits=(-4, np.percentile(prob_zarr, 99.8)))

    # map from points set to label masks
    pd_mask = np.zeros_like(prob_zarr[0])
    # pd_mask[:] = np.nan
    nucleus_id_u = np.unique(point_df["nucleus_id"])
    feature_cols = [col for col in point_df.columns if "feat" in col]
    point_df.loc[:, feature_cols] = point_df.loc[:, feature_cols] - np.mean(point_df.loc[:, feature_cols].to_numpy(), axis=0)
    point_df.loc[:, feature_cols] = np.divide(point_df.loc[:, feature_cols], np.std(point_df.loc[:, feature_cols], axis=0))
    regions = regionprops(mask_zarr)
    coord_list = [rg["coords"] for rg in regions]
    lb_list = np.asarray([rg["label"] for rg in regions])
    for n, nc_id in enumerate(tqdm(nucleus_id_u, "Assigning feature values...")):
        feat_val = point_df.loc[labels_df["nucleus_id"] == nc_id, feature_cols[feature_num]].values
        ind = np.where(lb_list == nc_id)[0][0]
        coords = coord_list[ind]
        pd_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = feat_val


    pd_layer = viewer.add_image(pd_mask, scale=scale_vec, name='prediction', colormap="inferno", opacity=0.65, visible=True)

    napari.run()
    # mlp_df.to_csv(mlp_data_path, index=False)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date = "20240620"# "20240712_01"
    overwrite = True
    fluo_flag = False
    use_model_priors = False
    seg_model = "tdTom-bright-log-v5" #"tdTom-bright-log-v5"  # "tdTom-dim-log-v3"
    # point_model = "point_models_pos"
    well_num = 12
    time_int = 6
    plot_model_features(root, experiment_date=experiment_date, well_num=well_num, seg_type="tissue_only_best_model_tissue", #seg_type="seg01_best_model_tbx5a", #
                    seg_model=seg_model, time_int=time_int, feature_num=25)

