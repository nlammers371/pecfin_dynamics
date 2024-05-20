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


def extract_nucleus_stats(root, rawRoot, level, experiment_date, model_name, fluo_channels, resetClassGB=False,
                          overwriteCellLabelsGB=False):
    # get directory to stitched labels
    mask_directory = os.path.join(root, "built_data", "stitched_labels", model_name, experiment_date, '')

    # raw data dir
    raw_directory = os.path.join(root, "built_image_data", "zarr_image_files", experiment_date, '')

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


        # extract useful info
        scale_vec = data_zarr.attrs["voxel_size_um"]

        for t in time_indices0:

            # add layer of mask centroids
            regions = regionprops(mask_zarr[t])

            centroid_array = np.empty((len(regions), 3))
            for rgi, rg in enumerate(regions):
                centroid_array[rgi, :] = np.multiply(rg.centroid, scale_vec)

            # convert centroid array to data frame
            nucleus_df = pd.DataFrame(centroid_array, columns=["Z", "Y", "X"])

            # add additional info
            area_vec = []
            for rgi, rg in enumerate(regions):
                area_vec.append(rg.area)
            nucleus_df["Area"] = area_vec



            ################################
            # Calculate mRNA levels in each nucleus

            image_array = data_zarr[t]

            raise Exception("Left off here")
            # compute each channel of image array separately to avoid dask error
            im_array_list = []
            for ch in mRNA_channels:
                im_temp = image_array[ch, :, :, :]
                im_array_list.append(im_temp)

            # initialize empty columns
            print("Estimating cellular mRNA levels")
            omero_attrs = image_node.root.zarr.root_attrs['omero']
            channel_metadata = omero_attrs['channels']  # list of channels and relevant info
            channel_names = [channel_metadata[i]["label"] for i in mRNA_channels]
            for ch in channel_names:
                nucleus_df[ch] = np.nan
                nucleus_df[ch + "_mean"] = np.nan

            # calculate mRNA levels
            for rgi, rg in enumerate(regions):
                # iterate through channels
                nc_coords = rg.coords.astype(int)
                n_pix = nc_coords.shape[0]
                for chi, im_ch in enumerate(im_array_list):
                    # nc_ch_coords = np.concatenate((np.ones((n_pix,1))*ch, nc_coords), axis=1).astype(int)
                    mRNA_int = np.sum(im_ch[tuple(nc_coords.T)])
                    nucleus_df.loc[rgi, channel_names[chi] + "_mean"] = mRNA_int / n_pix
                    nucleus_df.loc[rgi, channel_names[chi]] = mRNA_int

            # normalize
            for ch in channel_names:
                int_max = np.max(nucleus_df[ch])
                mean_max = np.max(nucleus_df[ch + "_mean"])
                nucleus_df[ch] = nucleus_df.loc[:, ch] / int_max
                nucleus_df[ch + "_mean"] = nucleus_df.loc[:, ch + "_mean"] / mean_max

            #######################################################
            # expand nuclear masks to define cellular neighborhoods

            if overwriteCellLabels:
                label_array_cell = calculate_cell_masks(label_array, labelPath, nucleus_df, image_data, scale_vec, level,
                                                        multiscale_attrs, dataset_info)
            else:
                reader_lb_cell = Reader(parse_url(cellLabelPath))

                # nodes may include images, labels etc
                nodes_lb_cell = list(reader_lb_cell())

                # first node will be the image pixel data
                label_node_cell = nodes_lb_cell[1]
                label_array_cell = label_node_cell.data[level]

                label_array_cell = label_array_cell.compute()

            # now repeat above mRNA inference, this time using cell neighborhoods
            for ch in channel_names:
                nucleus_df[ch + "_cell"] = np.nan
                nucleus_df[ch + "_cell_mean"] = np.nan

            # calculate mRNA levels
            cell_regions = regionprops(label_array_cell)
            for rgi, rg in enumerate(cell_regions):
                # iterate through channels
                nc_coords = rg.coords.astype(int)
                n_pix = nc_coords.shape[0]
                for chi, im_ch in enumerate(im_array_list):
                    # nc_ch_coords = np.concatenate((np.ones((n_pix,1))*ch, nc_coords), axis=1).astype(int)
                    mRNA_int = np.sum(im_ch[tuple(nc_coords.T)])
                    nucleus_df.loc[rgi, channel_names[chi] + "_cell_mean"] = mRNA_int / n_pix
                    nucleus_df.loc[rgi, channel_names[chi] + "_cell"] = mRNA_int

            for ch in channel_names:
                int_max = np.max(nucleus_df.loc[:, ch + "_cell"])
                mean_max = np.max(nucleus_df.loc[:, ch + "_cell_mean"])
                nucleus_df.loc[:, ch + "_cell"] = nucleus_df.loc[:, ch + "_cell"] / int_max
                nucleus_df.loc[:, ch + "_cell_mean"] = nucleus_df.loc[:, ch + "_cell_mean"] / mean_max

            # remove empty rows (why do these appear?)
            nan_flag = np.isnan(nucleus_df.loc[:, "X"])
            nucleus_df = nucleus_df.loc[~nan_flag, :]

            # Average across nearest neighbors for both nucleus- and cell-based mRNA estimates
            nn_k = 6
            tree = KDTree(nucleus_df[["Z", "Y", "X"]], leaf_size=2)
            nearest_dist, nearest_ind = tree.query(nucleus_df[["Z", "Y", "X"]], k=nn_k + 1)

            for ch in channel_names:
                nucleus_df[ch + "_cell_nn"] = np.nan
                nucleus_df[ch + "_cell_mean_nn"] = np.nan
                nucleus_df[ch + "_nn"] = np.nan
                nucleus_df[ch + "_mean_nn"] = np.nan

            for row in range(nucleus_df.shape[0]):
                nearest_indices = nearest_ind[row, :]
                nn_df_temp = nucleus_df.iloc[nearest_indices]

                for ch in channel_names:
                    int_mean = np.mean(nn_df_temp.loc[:, ch])
                    mean_mean = np.mean(nn_df_temp.loc[:, ch + "_mean"])
                    int_mean_cell = np.mean(nn_df_temp.loc[:, ch + "_cell"])
                    mean_mean_cell = np.mean(nn_df_temp.loc[:, ch + "_cell_mean"])

                    nucleus_df.loc[row, ch + "_cell_nn"] = mean_mean_cell
                    nucleus_df.loc[row, ch + "_cell_mean_nn"] = int_mean_cell
                    nucleus_df.loc[row, ch + "_nn"] = int_mean
                    nucleus_df.loc[row, ch + "_mean_nn"] = mean_mean

            nucleus_df.to_csv(savePath)


if __name__ == "__main__":
    # define some variables
    level = 0
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"
    rawRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/raw/"

    extract_nucleus_stats(dataRoot, rawRoot, level=0)
