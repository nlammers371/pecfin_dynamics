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
import math


def ellipsoid_axis_lengths(central_moments):
    """Compute ellipsoid major, intermediate and minor axis length.

    Parameters
    ----------
    central_moments : ndarray
        Array of central moments as given by ``moments_central`` with order 2.

    Returns
    -------
    axis_lengths: tuple of float
        The ellipsoid axis lengths in descending order.
    """
    m0 = central_moments[0, 0, 0]
    sxx = central_moments[2, 0, 0] / m0
    syy = central_moments[0, 2, 0] / m0
    szz = central_moments[0, 0, 2] / m0
    sxy = central_moments[1, 1, 0] / m0
    sxz = central_moments[1, 0, 1] / m0
    syz = central_moments[0, 1, 1] / m0
    S = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    # determine eigenvalues in descending order
    eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]
    eigvals[eigvals < 0] = 0
    return tuple([math.sqrt(5.0 * e) for e in eigvals])


def ellipsoid_axis_loadings(central_moments):
    """Compute ellipsoid major, intermediate and minor axis length.

    Parameters
    ----------
    central_moments : ndarray
        Array of central moments as given by ``moments_central`` with order 2.

    Returns
    -------
    axis_lengths: tuple of float
        The ellipsoid axis lengths in descending order.
    """
    m0 = central_moments[0, 0, 0]
    sxx = central_moments[2, 0, 0] / m0
    syy = central_moments[0, 2, 0] / m0
    szz = central_moments[0, 0, 2] / m0
    sxy = central_moments[1, 1, 0] / m0
    sxz = central_moments[1, 0, 1] / m0
    syz = central_moments[0, 1, 1] / m0
    S = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    # # determine eigenvalues in descending order
    eigvals, eigvecs = np.linalg.eig(S)
    eigpower = np.sum(np.multiply(np.asarray(eigvals)[np.newaxis, :]**2, np.asarray(np.abs(eigvecs))), axis=1)
    eigpower = eigpower / np.sum(np.abs(eigpower))
    return eigpower

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
            point_path = os.path.join(point_directory, point_prefix + "_points.csv")

            if (not os.path.isfile(point_path)) | overwrite_flag:
                # add layer of mask centroids
                # NL: note that this techincally should factor in pixel dims, but I've found that z distortion
                #     compromises shape measures
                regions = regionprops(mask_zarr[t])#, spacing=tuple(scale_vec))

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
                nucleus_df["size"] = np.asarray([rg.area for rg in regions])

                # remove very small masks
                size_filter = nucleus_df["size"].to_numpy() >= 15
                nucleus_df = nucleus_df.loc[size_filter, :]
                regions = [regions[i] for i in range(len(regions)) if size_filter[i]]

                # add nucleus shape info
                nucleus_df[["e0", "e1", "e2"]] = np.asarray(
                    [ellipsoid_axis_lengths(rg["moments_central"]) for rg in regions])
                p_array = np.asarray([ellipsoid_axis_loadings(rg["moments_central"]) for rg in regions])
                # if np.any(np.iscomplex(p_array)):
                #     print("sigh")
                nucleus_df[["pZ", "pY", "pX"]] = np.real(p_array)
                nucleus_df["eccentricity"] = np.divide(nucleus_df["e0"].to_numpy(), np.sqrt(
                    np.sum(nucleus_df.loc[:, ["e0", "e1", "e2"]].to_numpy() ** 2, axis=1)))  # 1 - np.divide(nucleus_df["size"], (1.33 * np.pi * nucleus_df["e0"]**3))
                nucleus_df["eccentricity"] = (nucleus_df["eccentricity"] - 1 / np.sqrt(3)) / (1 - 1 / np.sqrt(3))
                nucleus_df.loc[np.isnan(nucleus_df["eccentricity"]), "eccentricity"] = 0
                nucleus_df.loc[np.isnan(nucleus_df["eccentricity"]), ["pX", "pY", "pZ"]] = 0

                # perform NN smoothing
                nn_k = 5
                tree = KDTree(nucleus_df[["Z", "Y", "X"]], leaf_size=2)
                nearest_dist, nearest_ind = tree.query(nucleus_df[["Z", "Y", "X"]], k=nn_k + 1)

                ch_list = ["pZ", "pY", "pX", "size", "eccentricity"]
                for ch in ch_list:
                    p_array = nucleus_df.loc[:, ch].to_numpy()
                    p_array_nn = p_array[nearest_ind]
                    nucleus_df[ch + "_nn"] = np.nanmean(p_array_nn, axis=1)

                if (len(data_zarr.shape) == 5) & (fluo_channels is None):
                    channel_dims = data_zarr.shape[0]
                    nuclear_channel = data_zarr.attrs["nuclear_channel"]
                    fluo_channels = [ch for ch in range(channel_dims) if ch != nuclear_channel]
                    fluo_names = [data_zarr.attrs["channel_names"][f] for f in fluo_channels]

                elif fluo_channels is None:
                    fluo_names = []

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
                    # nn_k = 5
                    # tree = KDTree(nucleus_df[["Z", "Y", "X"]], leaf_size=2)
                    # nearest_dist, nearest_ind = tree.query(nucleus_df[["Z", "Y", "X"]], k=nn_k + 1)

                    for ch in fluo_names:
                        fluo_array = nucleus_df.loc[:, ch + "_mean"].to_numpy()
                        fluo_array_nn = fluo_array[nearest_ind]
                        nucleus_df[ch + "_mean_nn"] = np.nanmean(fluo_array_nn, axis=1)

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
                mean_cols = ["Z", "Y", "X", "pZ_nn", "pY_nn", "pX_nn", "size", "eccentricity"]
                fluo_cols = []
                for ch in fluo_names:
                    fluo_cols.append(ch + "_mean_nn")

                point_df = nucleus_df.loc[:, point_cols + mean_cols + fluo_cols].groupby(point_cols).mean(mean_cols + fluo_cols).reset_index()
                point_df.loc[:, fluo_cols] = point_df.loc[:, fluo_cols] - np.min(point_df.loc[:, fluo_cols], axis=0)
                point_df.loc[:, fluo_cols] = np.divide(point_df.loc[:, fluo_cols], np.max(point_df.loc[:, fluo_cols], axis=0))
                if np.any(np.isnan(point_df.loc[:, mean_cols])):
                    print("wtf")
                # save
                nucleus_df.to_csv(nucleus_path, index=False)
                point_df.to_csv(point_path, index=False)


if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    experiment_date_vec = ["20240223"] #["20240424", "20240425", "20240223"]
    seg_model_vec = ["log-v5"] #["log-v3", "log-v3", "log-v5"]
    # build point cloud files
    for e, experiment_date in enumerate(experiment_date_vec):
        extract_nucleus_stats(root, experiment_date, seg_model_vec[e], overwrite_flag=True)
