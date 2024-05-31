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
                point_cols = ["experiment_date", "seg_model", "well_num", "time_int", "cluster_id"]
                mean_cols = ["Z", "Y", "X"]
                fluo_cols = []
                for ch in fluo_names:
                    fluo_cols.append(ch + "_mean_nn")

                point_df = nucleus_df.loc[:, point_cols + mean_cols + fluo_cols].groupby(point_cols).mean(mean_cols + fluo_cols).reset_index()
                point_df.loc[:, fluo_cols] = point_df.loc[:, fluo_cols] - np.min(point_df.loc[:, fluo_cols], axis=0)
                point_df.loc[:, fluo_cols] = np.divide(point_df.loc[:, fluo_cols], np.max(point_df.loc[:, fluo_cols], axis=0))

                # save
                nucleus_df.to_csv(nucleus_path, index=False)
                point_df.to_csv(point_path, index=False)


if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    experiment_date_vec = ["20240424", "20240425", "20240223"]
    seg_model = "log-v3"
    # build point cloud files
    for experiment_date in experiment_date_vec:
        extract_nucleus_stats(root, experiment_date, seg_model, overwrite_flag=True)
