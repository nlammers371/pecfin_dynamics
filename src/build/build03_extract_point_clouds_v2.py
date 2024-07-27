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
from sklearn.preprocessing import KBinsDiscretizer

def extract_nucleus_stats(root, experiment_date, model_name, fluo_channels=None, overwrite_flag=False):

    point_directory = os.path.join(root, "point_cloud_data", "nucleus_point_clouds", "data")
    if not os.path.isdir(point_directory):
        os.makedirs(point_directory)

    fluo_directory = os.path.join(root, "point_cloud_data", "tbx5a_segmentation")
    if not os.path.isdir(fluo_directory):
        os.makedirs(fluo_directory)

    # get directory to stitched labels
    mask_directory = os.path.join(root, "built_data", "mask_stacks", model_name, experiment_date, '')

    # raw data dir
    raw_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')

    curation_path = os.path.join(root, "metadata", "curation", experiment_date + "_curation_info.csv")
    has_curation_info = os.path.isfile(curation_path)
    if has_curation_info:
        curation_df = pd.read_csv(curation_path)
        curation_df_long = pd.melt(curation_df,
                                   id_vars=["series_number", "notes", "tbx5a_flag", "follow_up_flag"],
                                   var_name="time_index", value_name="qc_flag")
        curation_df_long["time_index"] = curation_df_long["time_index"].astype(int)

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

        tbx5a_flag = 0
        if has_curation_info:
            tbx5a_flag = curation_df_long.loc[
                (curation_df_long.series_number == well_num), "tbx5a_flag"].to_numpy()[0]

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
                # NL: note that this technically should factor in pixel dims, but I've found that z distortion
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
                nucleus_df.reset_index(inplace=True, drop=True)
                # regions = [regions[i] for i in range(len(regions)) if size_filter[i]]

                if (len(data_zarr.shape) == 5) & (fluo_channels is None) & (tbx5a_flag == 1):
                    channel_dims = data_zarr.shape[0]
                    nuclear_channel = data_zarr.attrs["nuclear_channel"]
                    fluo_channels = [ch for ch in range(channel_dims) if ch != nuclear_channel]
                    fluo_names = [data_zarr.attrs["channel_names"][f] for f in fluo_channels]

                elif (fluo_channels is None) | (tbx5a_flag == 0):
                    fluo_names = []

                ################################
                # Calculate mRNA levels in each nucleus
                if (len(data_zarr.shape) == 5) & (fluo_channels is not None) & (tbx5a_flag == 1):
                    fluo_path = os.path.join(fluo_directory, point_prefix + "_labels.csv")
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
                    fluo_df = nucleus_df.copy()
                    for f, fluo_name in enumerate(fluo_names):
                        fluo_df.loc[:, fluo_name + "_mean"] = scipy.ndimage.mean(im_array_list[f], mask_zarr[t], fluo_df["nucleus_id"].to_numpy())

                    for ch in fluo_names:
                        fluo_array = fluo_df.loc[:, ch + "_mean"].to_numpy()
                        fluo_array_nn = fluo_array[nearest_ind]
                        fluo_df[ch + "_mean_nn"] = np.nanmean(fluo_array_nn, axis=1)

                    # normalize
                    for ch in fluo_names:
                        # int_max = np.max(nucleus_df[ch + "_nn"])
                        # mean_max = np.max(fluo_df[ch + "_mean_nn"])
                        # nucleus_df[ch + "_nn_norm"] = nucleus_df.loc[:, ch + "_nn"] / int_max
                        # fluo_df[ch + "_mean_nn_norm"] = fluo_df.loc[:, ch + "_mean_nn"] / mean_max
                        # disc0 = KBinsDiscretizer(n_bins=46, encode="ordinal", strategy="quantile")
                        # f_vec = fluo_df[ch + "_mean_nn_norm"].to_numpy()[:, np.newaxis]
                        # disc0.fit(f_vec)
                        # bin_vec0 = disc0.transform(f_vec)
                        # fluo_df[ch + "_fluo_label"] = bin_vec0
                        # calculate distance to max fluo value
                        arg_max = np.argmax(fluo_df[ch + "_mean_nn"])
                        max_zyx = fluo_df.loc[arg_max, ["Z", "Y", "X"]].to_numpy().astype(np.float64)

                        fluo_df[ch + "_dist"] = np.sqrt(np.sum((fluo_df.loc[:, ["Z", "Y", "X"]].to_numpy() - max_zyx)**2, axis=1))
                        # dist_norm = fluo_df[ch + "_dist"].to_numpy() / np.max(fluo_df[ch + "_dist"])
                        # disc1 = KBinsDiscretizer(n_bins=50, encode="ordinal", strategy="quantile")
                        # disc1.fit(dist_norm[:, np.newaxis])
                        # bin_vec1 = disc1.transform(dist_norm[:, np.newaxis])
                        # fluo_df[ch + "_dist_label"] = bin_vec1

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

                # save labeled dataset if we have fluo data
                if tbx5a_flag:
                    fluo_df.to_csv(fluo_path, index=False)


def generate_fluorescence_labels(fluo_df_path, fluo_var, nbins=21, bin_method="kmeans"):
    day2_offset = 75  # temporary fudge factor until I get reliable stage estimates

    df_list = sorted(glob.glob(fluo_df_path + "*.csv"))
    fluo_df_list = []
    for d, df_path in enumerate(tqdm(df_list, "Loading point datasets...")):
        df = pd.read_csv(df_path)
        if df.loc[0, "experiment_date"].astype(str) == "20240425":  # temporary adjustment factor
            df["time_int"] += day2_offset
        df["frame_id"] = d
        fluo_df_list.append(df)

    # generate master DF
    fluo_df_master = pd.concat(fluo_df_list, axis=0, ignore_index=True)

    # get max fluo for each stage/time point
    time_index = np.unique(fluo_df_master["time_int"])
    time_window = 2
    time_vec = fluo_df_master["time_int"].values
    fluo_vec = fluo_df_master[fluo_var].values

    f_99_vec = []
    for t in time_index:
        t_filter = (time_vec >= t - time_window) & (time_vec <= t + time_window)
        f_99_vec.append(np.percentile(fluo_vec[t_filter], 99.9))

    # generate new normalized variable
    for t in tqdm(time_index):
        f_factor = np.asarray(f_99_vec)[time_index == t]
        fluo_df_master.loc[time_vec == t, fluo_var + "_norm"] = fluo_df_master.loc[time_vec == t, fluo_var] / f_factor
    fluo_df_master.loc[fluo_df_master[fluo_var + "_norm"] > 1, fluo_var + "_norm"] = 1

    # assign to bins according to fluorescence
    fluo_norm_vec = fluo_df_master[fluo_var + "_norm"].to_numpy()[:, np.newaxis]
    est = KBinsDiscretizer(
        n_bins=nbins, encode='ordinal', strategy=bin_method
    )
    est.fit(fluo_norm_vec)

    # add to dataset
    fluo_df_master["fluo_label"] = est.transform(fluo_norm_vec)
    fluo_df_master["fluo_val_norm"] = fluo_norm_vec

    rm_cols = ['tbx5a-StayGold_fluo_label', 'tbx5a-StayGold_mean_nn_norm', 'fin_curation_flag']
    fluo_df_master.drop(rm_cols, axis=1, inplace=True)

    for d, df_path in enumerate(tqdm(df_list, "Loading point datasets...")):
        df_updated = fluo_df_master.loc[fluo_df_master["frame_id"]==d, :].drop(["frame_id"], axis=1).reset_index(drop=True)

        if df_updated.loc[0, "experiment_date"].astype(str) == "20240425":  # temporary adjustment factor
            df_updated["time_int"] -= day2_offset

        df_updated.to_csv(df_path, index=False)

def make_segmentation_training_folder(root):

    # make output directory
    out_dir = os.path.join(root, "point_cloud_data", "segmentation_training", "data", "")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load tissue part labels
    tissue_dir = os.path.join(root, "point_cloud_data", "fin_segmentation", "")
    tissue_df_list = sorted(glob.glob(os.path.join(tissue_dir, "*.csv")))
    for df_path in tqdm(tissue_df_list, desc="Writing tissue datasets..."):
        df = pd.read_csv(df_path)
        keep_cols = ["Z", "Y", "X", "experiment_date", "well_num", "time_int", "nucleus_id", "fin_label_final"]
        df = df.loc[:, keep_cols]
        df.rename(columns={"fin_label_final": "label"}, inplace=True)
        df["label"] = df["label"] - 1
        df["class"] = ["tissue"]*df.shape[0]
        df["class_id"] = [0] * df.shape[0]
        # save
        df_name = path_leaf(df_path).replace("_labels", "_tissue_labels")
        df.to_csv(os.path.join(out_dir, df_name), index=False)

    fluo_dir = os.path.join(root, "point_cloud_data", "tbx5a_segmentation", "")
    fluo_df_list = sorted(glob.glob(os.path.join(fluo_dir, "*.csv")))
    for df_path in tqdm(fluo_df_list, desc="Writing tbx5a datasets..."):
        df = pd.read_csv(df_path)
        keep_cols = ["Z", "Y", "X", "experiment_date", "well_num", "time_int", "nucleus_id", "fluo_label"]
        df = df.loc[:, keep_cols]
        df.rename(columns={"fluo_label": "label"}, inplace=True)
        df["label"] = df["label"] + 4
        df["class"] = ["tbx5a"] * df.shape[0]
        df["class_id"] = [1] * df.shape[0]
        # save
        df_name = path_leaf(df_path).replace("_labels", "_fluo_labels")
        df.to_csv(os.path.join(out_dir, df_name), index=False)






if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    experiment_date_vec = ["20240620"]  #["20240424", "20240425", "20240223"]
    seg_model_vec = ["tdTom-bright-log-v5"]  #["log-v3", "log-v3", "log-v5"]
    # build point cloud files
    for e, experiment_date in enumerate(experiment_date_vec):
        extract_nucleus_stats(root, experiment_date, seg_model_vec[e], overwrite_flag=True)
