import os
import numpy as np
import glob2 as glob
import pandas as pd
from setuptools.command.egg_info import overwrite_arg
from tqdm import tqdm
from skimage.measure import regionprops
import zarr
from src.utilities.functions import path_leaf
from sklearn.neighbors import KDTree
import scipy
from sklearn.preprocessing import KBinsDiscretizer
from src.utilities.fin_class_def import FinData
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

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
                size_filter = nucleus_df["size"].to_numpy() >= 50
                nucleus_df = nucleus_df.loc[size_filter, :]
                nucleus_df.reset_index(inplace=True, drop=True)
                # regions = [regions[i] for i in range(len(regions)) if size_filter[i]]

                if (len(data_zarr.shape) == 5) & (fluo_channels is None) & (tbx5a_flag == 1):
                    channel_dims = data_zarr.shape[0]
                    nuclear_channel = data_zarr.attrs["nuclear_channel"]
                    fluo_channels = [ch for ch in range(channel_dims) if ch != nuclear_channel]
                    fluo_names = [data_zarr.attrs["channel_names"][f] for f in fluo_channels]

                elif (len(data_zarr.shape) == 5) & (fluo_channels is not None) & (tbx5a_flag == 1):
                    fluo_names = [data_zarr.attrs["channel_names"][f] for f in fluo_channels]

                else:
                    tbx5a_flag = 0
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
                        if np.any(np.isnan(fluo_df[ch + "_mean_nn"])):
                            print("wtf")

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
                    if "tbx5a-StayGold_mean_nn" not in fluo_df.columns:
                        print("why?")
                    if np.any(np.isnan(fluo_df["tbx5a-StayGold_mean_nn"])):
                        print("wtf")
                    fluo_df.to_csv(fluo_path, index=False)


def generate_fluorescence_labels(fluo_df_path, fluo_var, nbins=21, bin_method="kmeans"):
    day2_offset = 75  # temporary fudge factor until I get reliable stage estimates

    df_list = sorted(glob.glob(fluo_df_path + "*.csv"))
    out_path = fluo_df_path[:-1] + "_lb"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    fluo_df_list = []
    for d, df_path in enumerate(tqdm(df_list, "Loading point datasets...")):
        try:
            df = pd.read_csv(df_path)
            if df.loc[0, "experiment_date"].astype(str) == "20240425":  # temporary adjustment factor
                df["time_int"] += day2_offset
            df["frame_id"] = d

            fluo_df_list.append(df)
        except:
            pass

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
    if np.any(np.isnan(fluo_norm_vec)):
        print("why")
    est = KBinsDiscretizer(
        n_bins=nbins, encode='ordinal', strategy=bin_method
    )
    est.fit(fluo_norm_vec)

    # add to dataset
    fluo_df_master["fluo_label"] = est.transform(fluo_norm_vec)
    fluo_df_master["fluo_val_norm"] = fluo_norm_vec

    rm_cols = ['tbx5a-StayGold_fluo_label', 'tbx5a-StayGold_mean_nn_norm', 'fin_curation_flag']
    rm_cols = [col for col in rm_cols if col in fluo_df_master.columns]
    if len(rm_cols) > 0:
        fluo_df_master.drop(rm_cols, axis=1, inplace=True)

    for d, df_path in enumerate(tqdm(df_list, "Saving point datasets...")):
        df_updated = fluo_df_master.loc[fluo_df_master["frame_id"] == d, :].drop(["frame_id"], axis=1).reset_index(drop=True)
        if df_updated.shape[0] > 10:
            if df_updated.loc[0, "experiment_date"].astype(str) == "20240425":  # temporary adjustment factor
                df_updated["time_int"] -= day2_offset

            df_name = path_leaf(df_path)
            out_name = os.path.join(out_path, df_name)
            df_updated.to_csv(out_name, index=False)

def make_segmentation_training_folder(root, out_suffix="", nucleus_size_thresh=150, fluo_share=1):

    # make output directory
    out_dir = os.path.join(root, "point_cloud_data", "segmentation_training" + out_suffix, "data", "")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load tissue part labels
    tissue_dir_list = glob.glob(os.path.join(root, "point_cloud_data", "fin_segmentation" + "*"))
    tissue_dir_list = [tdir for tdir in tissue_dir_list if os.path.isdir(tdir)]
    n_tissue = 0
    for t, tissue_dir in enumerate(tissue_dir_list):
        tissue_df_list = sorted(glob.glob(os.path.join(tissue_dir, "*.csv")))
        for df_path in tqdm(tissue_df_list, desc="Writing tissue datasets..."):
            df = pd.read_csv(df_path)
            df = df.loc[df["size"] > nucleus_size_thresh, :]
            if df.shape[0] > 4000:
                keep_cols = ["Z", "Y", "X", "experiment_date", "well_num", "time_int", "nucleus_id", "fin_label_final"]
                df = df.loc[:, keep_cols]
                df.rename(columns={"fin_label_final": "label"}, inplace=True)
                df["label"] = df["label"] - 1
                # df = df.loc[df["label"] < 3, :]  # remove outlier class

                df["class"] = ["tissue"]*df.shape[0]
                df["class_id"] = [0] * df.shape[0]
                # save
                df_name = path_leaf(df_path).replace("_labels", "_tissue_labels")
                df.to_csv(os.path.join(out_dir, df_name), index=False)

                n_tissue += 1

    fluo_dir_list = glob.glob(os.path.join(root, "point_cloud_data", "tbx5a_segmentation_lb" + "*"))
    fluo_dir_list = [fdir for fdir in fluo_dir_list if os.path.isdir(fdir)]
    n_fluo_max = fluo_share*n_tissue
    n_fluo = 0
    for f, fluo_dir in enumerate(fluo_dir_list):
        fluo_df_list = sorted(glob.glob(os.path.join(fluo_dir, "*.csv")))
        fluo_indices = np.random.choice(range(len(fluo_df_list)), size=len(fluo_df_list), replace=False)
        fluo_df_list = [fluo_df_list[f] for f in fluo_indices]
        for df_path in tqdm(fluo_df_list, desc="Writing tbx5a datasets..."):
            df = pd.read_csv(df_path)

            # df = df.loc[df["size"] > 100, :]
            if df.shape[0] > 4000:
                keep_cols = ["Z", "Y", "X", "experiment_date", "well_num", "time_int", "nucleus_id", "fluo_label"]
                df = df.loc[:, keep_cols]
                df.rename(columns={"fluo_label": "label"}, inplace=True)
                df["label"] = df["label"] + 4
                df["class"] = ["tbx5a"] * df.shape[0]
                df["class_id"] = [1] * df.shape[0]

                # save
                df_name = path_leaf(df_path).replace("_labels", "_fluo_labels")
                df.to_csv(os.path.join(out_dir, df_name), index=False)

                n_fluo += 1

            if n_fluo >= n_fluo_max:
                break



def make_vae_training_data(root, seg_model, overwrite_flag=False, out_suffix="", nucleus_size_thresh=100,
                           n_point_samples=2048, k_nn_thresh=3, min_points=25, scale_factor=25):

    # (0) get list of datasets
    #   -take from fin objects if available
    #   -else take from v1 of manual curation
    #   -else take from ML-generated labels
    # (1) filter for fin only
    # (2) apply QC (unless from fin object)
    # (3) Orient axes and upsample
    # (4) Save

    # make output directory
    out_dir = os.path.join(root, "point_cloud_data", "vae_training" + out_suffix, "data", "")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get comprehensive list of available training datasets
    point_feature_dir = os.path.join(root, "point_cloud_data", "point_features", seg_model, "")
    point_path_list = sorted(glob.glob(point_feature_dir + "*.csv"))
    orig_prefix_list = np.asarray([path_leaf(f).replace("_points_features.csv", "") for f in point_path_list])
    remaining_prefix_list = orig_prefix_list.copy()
    
    # generate master list of files to load
    master_path_list = []
    data_source_list = []
    master_prefix_list = []

    # check fin objects first
    fin_object_path = os.path.join(root, "point_cloud_data", "fin_objects", "")
    fin_object_list = sorted(glob.glob(fin_object_path + "*.pkl"))
    obj_prefix_list = np.asarray([path_leaf(fp).replace("_fin_object.pkl", "") for fp in fin_object_list])

    master_path_list += fin_object_list
    data_source_list += ["object"] * len(fin_object_list)
    master_prefix_list += list(obj_prefix_list)
    remaining_prefix_list = remaining_prefix_list[~np.isin(remaining_prefix_list, obj_prefix_list)]

    # check v1 manual curation next
    manual_dir_list = glob.glob(os.path.join(root, "point_cloud_data", "manual_curation", "" ) + "*")
    manual_dir_list = [tdir for tdir in manual_dir_list if os.path.isdir(tdir)]
    manual_path_list = []

    for t, tissue_dir in enumerate(manual_dir_list):
        manual_path_list += sorted(glob.glob(os.path.join(tissue_dir, "*.csv")))

    manual_path_list = np.asarray(manual_path_list)
    manual_prefix_list = np.asarray([path_leaf(fp).replace("_labels.csv", "") for fp in manual_path_list])
    manual_path_list = manual_path_list[np.isin(manual_prefix_list, remaining_prefix_list)]
    manual_prefix_list = manual_prefix_list[np.isin(manual_prefix_list, remaining_prefix_list)]

    master_path_list += list(manual_path_list)
    data_source_list += ["manual"] * len(manual_prefix_list)
    master_prefix_list += list(manual_prefix_list)
    remaining_prefix_list = remaining_prefix_list[~np.isin(remaining_prefix_list, manual_prefix_list)]

    # get the rest in original points folder
    point_ft = np.isin(orig_prefix_list, remaining_prefix_list)
    point_paths = np.asarray(point_path_list)[point_ft]

    master_path_list += list(point_paths)
    data_source_list += ["ml"] * len(point_paths)
    master_prefix_list += list(orig_prefix_list[point_ft])

    keep_cols = ["Z", "Y", "X", "experiment_date", "size", "seg_model", "data_source", "well_num", "time_int", "fin_label_curr"]

    ####
    # Iterate through paths and load data
    for d, df_path in enumerate(tqdm(master_path_list, "Saving point datasets...")):

        point_prefix = master_prefix_list[d]
        out_path = os.path.join(out_dir, point_prefix + ".csv")
        prev_flag = os.path.isfile(out_path)

        if not prev_flag or overwrite_flag:
            # determine correct protocol based off of datasource
            data_source = data_source_list[d]

            if data_source == "object":
                fin_data = FinData(data_root=root, name=point_prefix, tissue_seg_model=seg_model)
                fin_df = fin_data.full_point_data
                fin_df["data_source"] = "object"

            elif data_source == "manual":
                fin_df = pd.read_csv(df_path)
                fin_df["fin_label_curr"] = fin_df["fin_label_final"].copy()
                fin_df["data_source"] = "manual"

            elif data_source == "ml":
                fin_df = pd.read_csv(df_path)
                fin_df["fin_label_curr"] = fin_df["label_pd"].copy() + 1
                fin_df["data_source"] = "ml"

            else:
                raise Exception("Data source not recognized")

            # filter for only pec fin nuclei
            fin_df = fin_df.loc[fin_df["fin_label_curr"] == 1, keep_cols]
            fin_df.reset_index(inplace=True, drop=True)
            # apply basic QC filters
            # calculate NN using KD tree
            xyz_array = fin_df.loc[:, ["X", "Y", "Z"]].to_numpy()
            if xyz_array.shape[0] > min_points:
                tree = KDTree(xyz_array)
                nearest_dist, nearest_ind = tree.query(xyz_array, k=k_nn_thresh + 1)

                nn_mean = np.mean(nearest_dist, axis=0)
                nn_scale = nn_mean[1]
                space_outliers = (nearest_dist[:, k_nn_thresh] > 2 * nn_scale).ravel()

                #########
                # get size-based outliers
                size_outliers = (fin_df.loc[:, ["size"]] < nucleus_size_thresh).to_numpy().ravel()

                # update the data array
                outlier_flags = size_outliers | space_outliers
                fin_df = fin_df.loc[~outlier_flags, :] # set to outlier class
                fin_df.reset_index(inplace=True, drop=True)

                if fin_df.shape[0] > min_points:
                    xyz_array = fin_df.loc[:, ["X", "Y", "Z"]].to_numpy()
                    # orient fin relative to main axes
                    if data_source == "object" and point_prefix == fin_data.name:
                        fin_axes = fin_data.calculate_axis_array(fin_data.axis_fin)
                    else:
                        PCAFIN = PCA(n_components=3)
                        PCAFIN.fit(xyz_array)
                        fin_axes = PCAFIN.components_

                    pca_array = np.matmul(xyz_array - np.mean(xyz_array, axis=0), fin_axes.T)

                    # rescale and resample
                    pca_array = pca_array / scale_factor
                    # fit density
                    kde = KernelDensity(bandwidth=0.1, kernel="gaussian").fit(pca_array)

                    # draw samples
                    pca_array_rs = kde.sample(n_samples=n_point_samples)

                    # save
                    fin_df_out = pd.DataFrame(pca_array_rs, columns=["X", "Y", "Z"])
                    fin_df_out["class"] = "fin"
                    fin_df_out["class_id"] = 1
                    fin_df_out["experiment_date"] = fin_df.loc[0, "experiment_date"]
                    fin_df_out["well_num"] = fin_df.loc[0, "well_num"]
                    fin_df_out["time_int"] = fin_df.loc[0, "time_int"]
                    fin_df_out["point_prefix"] = point_prefix
                    fin_df_out["seg_model"] = fin_df.loc[0, "seg_model"]
                    fin_df_out["data_source"] = data_source

                    fin_df_out.to_csv(out_path, index=False)



if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    experiment_date_vec = ["20240620"]  #["20240424", "20240425", "20240223"]
    seg_model_vec = ["tdTom-bright-log-v5"]  #["log-v3", "log-v3", "log-v5"]
    # build point cloud files
    for e, experiment_date in enumerate(experiment_date_vec):
        extract_nucleus_stats(root, experiment_date, seg_model_vec[e], overwrite_flag=True)
