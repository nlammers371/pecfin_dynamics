import os
import numpy as np
import glob2 as glob
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.utilities.data_utilities import PointData
from src.point_net.point_net import PointNetSegHead
from skimage.measure import regionprops
import zarr
from src.utilities.functions import path_leaf
from sklearn.neighbors import KDTree
import scipy
from sklearn.cluster import KMeans

def extract_nucleus_stats(root, experiment_date, model_name, overwrite_flag=False, fluo_channels=None, n_clusters=4096):

    # nucleus data directory
    nucleus_directory = os.path.join(root, "built_data", "nucleus_data", "raw_nuclei", experiment_date,  '')
    if not os.path.isdir(nucleus_directory):
        os.makedirs(nucleus_directory)

    point_directory = os.path.join(root, "built_data", "nucleus_data", "point_clouds", experiment_date, '')
    if not os.path.isdir(point_directory):
        os.makedirs(point_directory)

    # get directory to stitched labels
    mask_directory = os.path.join(root, "built_data", "stitched_labels", model_name, experiment_date, '')

    # raw data dir
    raw_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')

    # load curation data if we have it
    curation_path = os.path.join(root, "metadata", "curation", experiment_date + "_curation_info.csv")
    has_curation_info = os.path.isfile(curation_path)
    if has_curation_info:
        curation_df = pd.read_csv(curation_path)
        curation_df_long = pd.melt(curation_df,
                                   id_vars=["series_number", "notes", "tbx5a_flag", "follow_up_flag"],
                                   var_name="time_index", value_name="qc_flag")
        curation_df_long["time_index"] = curation_df_long["time_index"].astype(int)


    # get list of wells with labels to stitch
    well_list = sorted(glob.glob(mask_directory + "*_labels_stitched.zarr"))

    for well in tqdm(well_list, "Extracting nucleus positions..."):

        # get well index
        well_index = well.find("_well")
        well_num = int(well[well_index+5:well_index+9])

        # load zarr files
        mask_zarr = zarr.open(well, mode='r')

        data_zarr_name = path_leaf(well).replace("_labels_stitched", "")
        data_zarr = zarr.open(os.path.join(raw_directory, data_zarr_name), mode='r')

        # get number of time points
        if has_curation_info:
            time_indices0 = curation_df_long.loc[
                (curation_df_long.series_number == well_num) & (curation_df_long.qc_flag == 1), "time_index"].to_numpy()
        else:
            time_indices0 = np.arange(mask_zarr.shape[0])

        indices_to_process = []
        for t in time_indices0:
            nz_flag = np.any(mask_zarr[t, :, :, :] != 0)
            if nz_flag:
                indices_to_process.append(t)

        # extract useful info
        try:
            scale_vec = data_zarr.attrs["voxel_size_um"]
        except:
            scale_vec = np.asarray([2.0, 0.55, 0.55])

        for t in tqdm(indices_to_process, "Processing time points..."):

            point_prefix = experiment_date + f"_well{well_num:04}" + f"_time{t:04}"
            nucleus_path = os.path.join(nucleus_directory, point_prefix + "_nuclei.csv")
            point_path = os.path.join(point_directory, point_prefix + "_nuclei.csv")

            if (not os.path.isfile(point_path)) | overwrite_flag:
                # add layer of mask centroids
                regions = regionprops(mask_zarr[t])

                centroid_array = np.asarray([rg["Centroid"] for rg in regions])
                centroid_array = np.multiply(centroid_array, scale_vec)

                # convert to dataframe
                nucleus_df = pd.DataFrame(centroid_array, columns=["Z", "Y", "X"])

                # add metadata
                nucleus_df["experiment_date"] = experiment_date
                nucleus_df["seg_model"] = model_name
                nucleus_df["well_num"] = well_num
                nucleus_df["time_int"] = t
                nucleus_df["fin_curation_flag"] = False

                # add additional info
                nucleus_df["nucleus_id"] = np.asarray([rg.label for rg in regions])
                nucleus_df["area"] = np.asarray([rg.area for rg in regions])

                if (len(data_zarr.shape) == 5) & (fluo_channels is None):
                    channel_dims = data_zarr.shape[0]
                    nuclear_channel = data_zarr.attrs["nuclear_channel"]
                    fluo_channels = [ch for ch in range(channel_dims) if ch != nuclear_channel]
                    fluo_names = [data_zarr.attrs["channel_names"][f] for f in fluo_channels]

                ################################
                # Calculate mRNA levels in each nucleus
                if (len(data_zarr.shape) == 5) & (fluo_channels is not None):

                    image_array = np.squeeze(data_zarr[:, t, :, :, :])

                    # compute each channel of image array separately to avoid dask error
                    im_array_list = []
                    for ch in fluo_channels:
                        im_temp = image_array[ch, :, :, :]
                        im_array_list.append(im_temp)

                    # get mean fluorescence
                    for f, fluo_name in enumerate(fluo_names):
                        nucleus_df.loc[:, fluo_name + "_mean"] = scipy.ndimage.mean(im_array_list[f], mask_zarr[t], nucleus_df["nucleus_id"].to_numpy())

                    # Average across nearest neighbors for both nucleus- and cell-based mRNA estimates
                    nn_k = 5
                    tree = KDTree(nucleus_df[["Z", "Y", "X"]], leaf_size=2)
                    nearest_dist, nearest_ind = tree.query(nucleus_df[["Z", "Y", "X"]], k=nn_k + 1)

                    for ch in fluo_names:
                        fluo_array = nucleus_df.loc[:, ch + "_mean"].to_numpy()
                        fluo_array_nn = fluo_array[nearest_ind]
                        nucleus_df[ch + "_mean_nn"] = np.mean(fluo_array_nn, axis=1)

                    # normalize
                    for ch in fluo_names:
                        # int_max = np.max(nucleus_df[ch + "_nn"])
                        mean_max = np.max(nucleus_df[ch + "_mean_nn"])
                        # nucleus_df[ch + "_nn_norm"] = nucleus_df.loc[:, ch + "_nn"] / int_max
                        nucleus_df[ch + "_mean_nn_norm"] = nucleus_df.loc[:, ch + "_mean_nn"] / mean_max

                        # calculate distance to max fluo value
                        arg_max = np.argmax(nucleus_df[ch + "_mean_nn"])
                        max_zyx = nucleus_df.loc[arg_max, ["Z", "Y", "X"]].to_numpy().astype(np.float64)
                        nucleus_df[ch + "_dist"] = np.sqrt(np.sum((nucleus_df.loc[:, ["Z", "Y", "X"]].to_numpy() - max_zyx)**2, axis=1))

                # use k-means clustering to obtain standardized
                n_clusters_emb = np.min([nucleus_df.shape[0], n_clusters])
                kmeans = KMeans(n_clusters=n_clusters_emb, random_state=0, n_init="auto").fit(nucleus_df.loc[:, ["Z", "Y", "X"]])
                nucleus_df.loc[:, "cluster_id"] = kmeans.labels_

                # make separate point DF
                point_cols = ["experiment_date", "seg_model", "well_num", "time_int", "cluster_id", "Z", "Y", "X"]
                point_df = nucleus_df.loc[:, point_cols].groupby(point_cols[:-3]).mean(["Z", "Y", "X"]).reset_index()

                # save
                nucleus_df.to_csv(nucleus_path, index=False)
                point_df.to_csv(point_path, index=False)

# def labels_to_point_cloud(root, experiment_date, seg_model, well_num, time_int, mask, scale_vec, overwrite=False):
#
#     # check if file exists
#     point_prefix = experiment_date + f"_well{well_num:04}" + f"_time{time_int:04}"
#     point_path = os.path.join(root, "built_data", "nucleus_data", experiment_date, seg_model,"")
#     if not os.path.isdir(point_path):
#         os.makedirs(point_path)
#
#     if (not os.path.isfile((point_path + point_prefix + "_nuclei.csv"))) | overwrite:
#         # use regionprops to get centroids for each nucleus label
#         regions = regionprops(mask)
#
#         if len(regions) > 0:
#             centroid_array = np.asarray([rg["Centroid"] for rg in regions])
#             centroid_array = np.multiply(centroid_array, scale_vec)
#
#             # convert to dataframe
#             point_df = pd.DataFrame(centroid_array, columns=["Z", "Y", "X"])
#
#             # add metadata
#             point_df["experiment_date"] = experiment_date
#             point_df["seg_model"] = seg_model
#             point_df["well_num"] = well_num
#             point_df["time_int"] = time_int
#             point_df["fin_curation_flag"] = False
#
#             # save
#             point_df.to_csv(point_path + point_prefix + "_nuclei.csv", index=False)
#
#         else:
#             point_df = []
#     else:
#         point_df = pd.read_csv(point_path + point_prefix + "_nuclei.csv")
#
#     return point_df

# def point_cloud_wrapper(root, experiment_date, seg_model, scale_vec, overwrite=False, suffix_string="_stitched.zarr"):
#
#     # get list of zarr mask files
#     mask_dir = os.path.join(root, "built_data", "stitched_labels", seg_model, experiment_date, "")
#     mask_file_list = sorted(glob.glob(mask_dir + "*" + suffix_string))
#
#     for mask_file in tqdm(mask_file_list, "Extracting nucleus information..."):
#
#         mask_zarr = zarr.open(mask_file, mode="r")
#         n_time_points = mask_zarr.shape[0]
#         indices_to_process = []
#         for t in range(n_time_points):
#             nz_flag = np.any(mask_zarr[t, :, :, :] != 0)
#             if nz_flag:
#                 indices_to_process.append(t)
#
#         mask_name = path_leaf(mask_file)
#         ind = mask_name.find("well")
#         well_num = int(mask_name[ind + 4:ind + 8])
#
#         for time_int in tqdm(indices_to_process):
#             _ = labels_to_point_cloud(root, experiment_date, seg_model, well_num, time_int, mask_zarr[time_int], scale_vec, overwrite)


def extract_point_cloud_features(root, model_root, model_name, experiment_date,
                                 fluo_channel=None, overwrite_flag=False):

    outpath = os.path.join(root, "built_data", "nucleus_data", "processed_point_clouds", experiment_date, "")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    model_path = os.path.join(model_root, model_name)
    # data_root = os.path.join(root,  "built_data", "point_clouds", "_Archive", "")
    data_root = os.path.join(root, "built_data", "nucleus_data", "point_clouds", experiment_date, "")

    # feature selection hyperparameters
    NUM_TEST_POINTS = 4096
    BATCH_SIZE = 16
    NUM_CLASSES = 14
    n_local = 64

    # generate dataloader to load fin point clouds
    point_data = PointData(data_root, split='test', npoints=NUM_TEST_POINTS, fluo_channel=fluo_channel)
    dataloader = DataLoader(point_data, batch_size=BATCH_SIZE, shuffle=True)

    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load pretrained model
    model = PointNetSegHead(num_points=NUM_TEST_POINTS, m=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # apply to FOVs to generate training features
    for batch_i, batch in enumerate(tqdm(dataloader, "Extracting point features...")):
        # extract data
        points = batch["data"]
        points = torch.transpose(points, 1, 2)
        raw_path = batch["path"]
        sample_indices = batch["point_indices"]

        point_path_vec = []
        global_path_vec = []
        new_indices = []
        for p, path in enumerate(raw_path):


            path_suffix = path_leaf(path)
            point_path = os.path.join(outpath, path_suffix)
            global_name = path_suffix.replace(".csv", "_global.csv")
            global_path = os.path.join(outpath, global_name)
            if not os.path.isfile(point_path):
                new_indices.append(p)

            point_path_vec.append(point_path)
            global_path_vec.append(global_path)

        if not overwrite_flag:
            new_indices = np.asarray(new_indices)
            points = points[new_indices]
            raw_path = [raw_path[i] for i in new_indices]
            point_path_vec = [point_path_vec[i] for i in new_indices]
            global_path_vec = [global_path_vec[i] for i in new_indices]
            sample_indices = sample_indices[new_indices]

        # pass points to model
        if len(raw_path) > 0:
            points = points.to(device)
            backbone = model.backbone
            pointfeat, _, _ = backbone(points)

            # now, load point data frames and add point features
            col_names_local = [f"feat_loc{f:04}" for f in range(n_local)]
            col_names_global = [f"feat_glob{f:04}" for f in range(pointfeat.shape[1] - n_local)]
            # col_names = col_names_local + col_names_global

            for p, path in enumerate(raw_path):
                # load DF
                point_df = pd.read_csv(path)

                # extract features
                features = np.asarray(np.squeeze(pointfeat[p, :, :]).detach().cpu().T)

                indices_raw = sample_indices[p, :]
                indices, ia = np.unique(indices_raw, return_index=True)
                # assign features to dataframe
                # pt = np.squeeze(points_raw[p, :, :])
                point_df.loc[indices, col_names_local] = features[ia, :n_local]

                # make global df
                global_df = pd.DataFrame(np.reshape(features[0, n_local:], (1, len(col_names_global))), columns=col_names_global)

                point_df.to_csv(point_path_vec[p], index=False)
                global_df.to_csv(global_path_vec[p], index=False)

if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    model_root = "/home/nick/projects/pecfin_dynamics/src/point_net/trained_models/"
    model_name = "seg_focal_dice_iou_rot/seg_model_89.pth"

    experiment_date_vec = ["20240223"]
    seg_model = "log-v5"
    # build point cloud files
    for experiment_date in experiment_date_vec:
        extract_nucleus_stats(root, experiment_date, seg_model, overwrite_flag=True)

        extract_point_cloud_features(root, model_root, model_name, experiment_date=experiment_date, overwrite_flag=True)