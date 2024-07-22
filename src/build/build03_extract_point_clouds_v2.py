import os
import numpy as np
import glob2 as glob
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops
import zarr
from src.utilities.functions import path_leaf
from sklearn.neighbors import KDTree
import scipy

def extract_nucleus_stats(root, experiment_date, model_name, fluo_channels=None, overwrite_flag=False):

    point_directory = os.path.join(root, "point_cloud_data", "nucleus_point_clouds", "data")
    if not os.path.isdir(point_directory):
        os.makedirs(point_directory)

    # get directory to stitched labels
    mask_directory = os.path.join(root, "built_data", "mask_stacks", model_name, experiment_date, '')

    # raw data dir
    raw_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')

    # get list of wells with labels to stitch
    well_list = sorted(glob.glob(mask_directory + "*_mask_aff.zarr"))

    for well in tqdm(well_list, "Extracting nucleus positions..."):

        # get well index
        well_index = well.find("_well")
        well_num = int(well[well_index+5:well_index+9])

        # load zarr files
        mask_zarr = zarr.open(well, mode='r')

        data_zarr_name = path_leaf(well).replace("_mask_aff", "")
        data_zarr = zarr.open(os.path.join(raw_directory, data_zarr_name), mode='r')

        time_indices0 = np.arange(mask_zarr.shape[0])

        indices_to_process = []
        for t in time_indices0:
            nz_flag = np.any(mask_zarr[t, :, :, :] != 0)
            if nz_flag:
                indices_to_process.append(t)

        # extract useful info
        scale_vec = mask_zarr.attrs["voxel_size_um"]

        for t in tqdm(indices_to_process, "Processing time points..."):

            point_prefix = experiment_date + f"_well{well_num:04}" + f"_time{t:04}"
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
                # regions = [regions[i] for i in range(len(regions)) if size_filter[i]]

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
                    # perform NN smoothing
                    nn_k = 5
                    tree = KDTree(nucleus_df[["Z", "Y", "X"]], leaf_size=2)
                    nearest_dist, nearest_ind = tree.query(nucleus_df[["Z", "Y", "X"]], k=nn_k + 1)

                    image_array = np.squeeze(data_zarr[:, t, :, :, :])

                    # compute each channel of image array separately to avoid dask error
                    im_array_list = []
                    for ch in fluo_channels:
                        im_temp = image_array[ch, :, :, :]
                        im_array_list.append(im_temp)

                    # get mean fluorescence
                    for f, fluo_name in enumerate(fluo_names):
                        nucleus_df.loc[:, fluo_name + "_mean"] = scipy.ndimage.mean(im_array_list[f], mask_zarr[t], nucleus_df["nucleus_id"].to_numpy())

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
                # n_clusters_emb = np.min([nucleus_df.shape[0], n_clusters])
                # kmeans = KMeans(n_clusters=n_clusters_emb, random_state=0, n_init="auto").fit(nucleus_df.loc[:, ["Z", "Y", "X"]])
                # nucleus_df.loc[:, "cluster_id"] = kmeans.labels_

                # make separate point DF
                # point_cols = ["experiment_date", "seg_model", "well_num", "time_int", "cluster_id"]
                # mean_cols = ["Z", "Y", "X", "pZ_nn", "pY_nn", "pX_nn", "size", "eccentricity"]
                # fluo_cols = []
                # for ch in fluo_names:
                #     fluo_cols.append(ch + "_mean_nn")
                #
                # point_df = nucleus_df.loc[:, point_cols + mean_cols + fluo_cols].groupby(point_cols).mean(mean_cols + fluo_cols).reset_index()
                # point_df.loc[:, fluo_cols] = point_df.loc[:, fluo_cols] - np.min(point_df.loc[:, fluo_cols], axis=0)
                # point_df.loc[:, fluo_cols] = np.divide(point_df.loc[:, fluo_cols], np.max(point_df.loc[:, fluo_cols], axis=0))
                # if np.any(np.isnan(point_df.loc[:, mean_cols])):
                #     print("wtf")
                # save
                nucleus_df.to_csv(point_path, index=False)


if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    experiment_date_vec = ["20240620"]  #["20240424", "20240425", "20240223"]
    seg_model_vec = ["tdTom-bright-log-v5"]  #["log-v3", "log-v3", "log-v5"]
    # build point cloud files
    for e, experiment_date in enumerate(experiment_date_vec):
        extract_nucleus_stats(root, experiment_date, seg_model_vec[e], overwrite_flag=True)
