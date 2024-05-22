import plotly.express as px
from ome_zarr.io import parse_url
import shutil
from ome_zarr.reader import Reader
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob2 as glob
from skimage.measure import label, regionprops, regionprops_table
from scipy.ndimage import distance_transform_edt
from aicsimageio import AICSImage
import math
import dask
import os.path
import zarr
import dask.array as da
from skimage.transform import resize
from sklearn.neighbors import KDTree
from src.utilities.functions import path_leaf


def extract_nucleus_stats(root, experiment_date, model_name, fluo_channels=None):

    # nucleus data directory
    nucleus_directory = os.path.join(root, "built_data", "nucleus_data", model_name, experiment_date,  '')
    if not os.path.isdir(nucleus_directory):
        os.makedirs(nucleus_directory)

    # get directory to stitched labels
    mask_directory = os.path.join(root, "built_data", "stitched_labels", model_name, experiment_date, '')

    # raw data dir
    raw_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')

    # load cueration data if we have it
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

    for well in tqdm(well_list, "Extracting fluorescence levels..."):

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
        scale_vec = data_zarr.attrs["voxel_size_um"]

        for t in indices_to_process:

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
            nucleus_df["Area"] = np.asarray([rg.area for rg in regions])

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

                # initialize empty columns
                for ch in fluo_names:
                    nucleus_df[ch] = np.nan
                    nucleus_df[ch + "_mean"] = np.nan

                # calculate mRNA levels
                for rgi, rg in enumerate(regions):
                    # iterate through channels
                    nc_coords = rg.coords.astype(int)
                    n_pix = nc_coords.shape[0]
                    for chi, im_ch in enumerate(im_array_list):
                        # nc_ch_coords = np.concatenate((np.ones((n_pix,1))*ch, nc_coords), axis=1).astype(int)
                        fluo_int = np.sum(im_ch[tuple(nc_coords.T)])
                        nucleus_df.loc[rgi, fluo_names[chi] + "_mean"] = fluo_int / n_pix
                        nucleus_df.loc[rgi, fluo_names[chi]] = fluo_int

                # Average across nearest neighbors for both nucleus- and cell-based mRNA estimates
                nn_k = 5
                tree = KDTree(nucleus_df[["Z", "Y", "X"]], leaf_size=2)
                nearest_dist, nearest_ind = tree.query(nucleus_df[["Z", "Y", "X"]], k=nn_k + 1)

                for ch in fluo_names:
                    nucleus_df[ch + "_nn"] = np.nan
                    nucleus_df[ch + "_mean_nn"] = np.nan

                for row in range(nucleus_df.shape[0]):
                    nearest_indices = nearest_ind[row, :]
                    nn_df_temp = nucleus_df.iloc[nearest_indices]

                    for ch in fluo_names:
                        int_mean = np.mean(nn_df_temp.loc[:, ch])
                        mean_mean = np.mean(nn_df_temp.loc[:, ch + "_mean"])
                        nucleus_df.loc[row, ch + "_nn"] = int_mean
                        nucleus_df.loc[row, ch + "_mean_nn"] = mean_mean

            # save
            point_prefix = experiment_date + f"_well{well_num:04}" + f"_time{t:04}"
            nucleus_df.to_csv(os.path.join(nucleus_directory, point_prefix + "_nuclei.csv"), index=False)


if __name__ == "__main__":

    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    model_name = "log-v3"
    experiment_data = "20240424"
    extract_nucleus_stats(root, experiment_data, model_name)
