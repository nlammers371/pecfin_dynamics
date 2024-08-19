# from aicsimageio import AICSImage
import numpy as np
import napari
import os
from src.utilities.functions import path_leaf
from glob2 import glob
import skimage.io as skio
import pandas as pd
# import open3d as o3d
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
import time
import vispy.color
import zarr
from sklearn.neighbors import KDTree
import networkx as nx
from src.utilities.point_cloud_utils import farthest_point_sample

def sample_reference_points(mlp_df, labels_df, point_df, npoints=50):

    labels_u, labels_to = np.unique(labels_df.loc[:, "axis_label_pd"], return_inverse=True)
    labels_per = np.ceil(npoints / len(labels_u)).astype(int)
    ref_indices = []
    for l, lb in enumerate(labels_u):
        lbi = np.where(labels_u == lb)[0]
        lb_indices = np.where(labels_to == lbi)[0]
        ns = np.min([labels_per, len(lb_indices)])
        if ns > 0:
            # _, temp_indices = farthest_point_sample(labels_df.loc[lb_indices, ["Z", "Y", "X"]].to_numpy(), ns)
            temp_indices = np.random.choice(lb_indices, size=ns, replace=False)
            ref_indices += temp_indices.tolist()

    # transfer labels
    labels_df.loc[ref_indices, "axis_label_curr"] = labels_df.loc[ref_indices, "axis_label_pd"]

    # add to training set
    mlp_df_temp = point_df.loc[labels_df["axis_label_curr"] != 0]
    mlp_df_temp.loc[:, "axis_label_curr"] = labels_df.loc[labels_df["axis_label_curr"] != 0, "axis_label_curr"].copy()

    if len(mlp_df) > 0:
        mlp_df = pd.concat([mlp_df, mlp_df_temp])
        mlp_df = mlp_df.drop_duplicates(keep="first", subset=["experiment_date", "well_num", "time_int", "nucleus_id"], ignore_index=True)
    else:
        mlp_df = mlp_df_temp.copy()

    return mlp_df, labels_df

def strip_dummy_cols(df):
    cols = df.columns
    keep_cols = [col for col in cols if "Unnamed" not in col]
    df = df[keep_cols]
    return df

def fit_mlp(labels_df, mdl, mlp_df):

    feature_cols = []
    feature_cols += [c for c in mlp_df.columns if "feat" in c] + ["fin_label_pd"]  # + ["well_num", "time_int", "date_norm"]
    X_train = mlp_df.loc[:, feature_cols]

    Y_train = mlp_df.loc[:, "axis_label_curr"].to_numpy()

    print("Updating axis predictions...")
    mdl = mdl.fit(X_train, Y_train)

    # get new predictions
    X_pd = point_df.loc[:, feature_cols]

    Y_pd = mdl.predict(X_pd)

    if labels_df is not None:
        labels_df.loc[:, "axis_label_pd"] = Y_pd
    else:
        pass

    return labels_df, Y_pd, mdl

def update_mlp_data(labels_df, mlp_df, point_df, intra_well_only):

    # generate wide feature DF for classifier training
    mlp_df_temp = point_df.loc[labels_df["axis_label_curr"] != 0]
    mlp_df_temp.loc[:, "axis_label_curr"] = labels_df.loc[labels_df["axis_label_curr"] != 0, "axis_label_curr"].copy()
    mlp_df_temp["fin_label_pd"] = labels_df["fin_label_pd"]

    well_num = point_df.loc[0, "well_num"]
    # remove observations from different wells if desired
    if intra_well_only and (len(mlp_df) > 0):
        mlp_df = mlp_df.loc[mlp_df["well_num"] == well_num]

    if len(mlp_df) > 0:
        mlp_df = pd.concat([mlp_df_temp, mlp_df])
        mlp_df = mlp_df.drop_duplicates(keep="first", subset=["experiment_date", "well_num", "time_int", "nucleus_id"],
                                        ignore_index=True)
    else:
        mlp_df = mlp_df_temp.copy()

    mlp_df = strip_dummy_cols(mlp_df)
    mlp_df = mlp_df.dropna()

    return mlp_df

def load_mlp_data_v2(root, mlp_arch, n_points_per_set=400, intra_well_only=False, fin_flag=False, class_split=None):

    point_path = os.path.join(root, "point_cloud_data", "point_features" + "_" + seg_type_global, "")
    if intra_well_only:
        n_points_per_set = 400

    labeled_point_path = os.path.join(root, "point_cloud_data", "fin_segmentation" + "_" + seg_type_global, "")
    if class_split is None:
        # class_split = np.asarray([0.32, 0.32, 0.32, 0.04])
        class_ref_array = np.asarray([1, 2, 3])
        class_split = 1 / len(class_ref_array) * np.ones_like(class_ref_array)

    class_split_int = (n_points_per_set * class_split).astype(int)

    # get list of extant labeled datasets
    df_list = glob(os.path.join(labeled_point_path, "*.csv"))

    # load each in and sample
    df_out = []
    if len(df_list) > 0:
        for df_path in df_list:
            # load labels
            lb_df = pd.read_csv(df_path)

            # load points
            point_name = path_leaf(df_path).replace("labels", "points_features")
            point_df = pd.read_csv(os.path.join(point_path, point_name))

            if "axis_label_final" in lb_df.columns:
                if np.all(~np.isnan(lb_df["axis_label_final"].to_numpy())):
                    class_u, label_map = np.unique(lb_df["axis_label_final"], return_inverse=True)

                    train_indices = []
                    for ind, lb in enumerate(class_u):
                        options = np.where(label_map == ind)[0]
                        lb_indices = np.random.choice(options, class_split_int[class_ref_array==lb], replace=True)
                        train_indices.extend(lb_indices)

                    lb_df_temp = lb_df.loc[train_indices, ["nucleus_id", "axis_label_final", "fin_label_pd"]]
                    point_df_temp = point_df.iloc[train_indices]
                    point_df_temp = point_df_temp.merge(lb_df_temp, how="left", on="nucleus_id")
                    df_out.append(point_df_temp)

        if len(df_out) > 0:
            df_out = pd.concat(df_out, axis=0, ignore_index=True)
            df_out["axis_label_curr"] = df_out["axis_label_final"]
            df_out.drop(["axis_label_final"], axis=1, inplace=True)

    # initialize model
    mdl = MLPClassifier(max_iter=5000, hidden_layer_sizes=mlp_arch)

    return df_out, mdl

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
    point_path_out = os.path.join(root, "point_cloud_data", "fin_segmentation" + "_" + seg_type_global, "")

    if not os.path.isdir(point_path_out):
        os.makedirs(point_path_out)

    point_df_temp = pd.read_csv(point_path + point_prefix + "_points_features.csv")
    point_df_temp = strip_dummy_cols(point_df_temp)
    point_df = point_df_temp.copy()

    # check for pre-existing labels DF
    if os.path.isfile(point_path_out + point_prefix + "_labels.csv"):
        labels_df = pd.read_csv(point_path_out + point_prefix + "_labels.csv")
        labels_df = strip_dummy_cols(labels_df)

        if "axis_label_curr" not in labels_df.columns:
            labels_df["axis_label_curr"] = 0
            labels_df["axis_label_pd"] = 0
            labels_df["axis_label_final"] = np.nan

    else:
        keep_cols = [col for col in point_df.columns if "feat" not in col]
        labels_df = point_df.loc[:, keep_cols]
        # labels_df["fin_curation_flag"] = False
        # labels_df["fin_curation_date"] = np.nan
        labels_df["axis_label_curr"] = 0

    labels_df.loc[labels_df["axis_label_curr"] == -1, "axis_label_curr"] = 0

    labels_df["axis_label_pd"] = labels_df["axis_label_curr"]

    return point_df, labels_df, point_prefix, point_path_out

def label_update_function(event):
    global mlp_df, mdl, train_counter

    if event.type == 'paint':

        # get event coordinates and values
        zv = event.value[0][0][0]
        yv = event.value[0][0][1]
        xv = event.value[0][0][2]
        for i in range(1, len(event.value)):
            zv = np.concatenate((zv, event.value[i][0][0]))
            yv = np.concatenate((yv, event.value[i][0][1]))
            xv = np.concatenate((xv, event.value[i][0][2]))
        mask_coords = tuple([zv, yv, xv])
        nc_ids = mask_zarr[mask_coords]
        ft = nc_ids != 0

        mask_coords = list(mask_coords)
        mask_coords[0] = mask_coords[0][ft]
        mask_coords[1] = mask_coords[1][ft]
        mask_coords[2] = mask_coords[2][ft]
        mask_coords = tuple(mask_coords)

        if len(mask_coords) > 0:
            nucleus_id = np.unique(mask_zarr[mask_coords])
            new_value = lb_layer.data[mask_coords][0]

            # update layer
            lb_layer.data[np.isin(mask_zarr, nucleus_id)] = new_value
            labels_df.loc[np.isin(labels_df["nucleus_id"], nucleus_id), "axis_label_curr"] = new_value

            # updated training DF
            mlp_df = point_df.loc[labels_df["axis_label_curr"] != 0]
            mlp_df.reset_index(inplace=True, drop=True)
            mlp_df.loc[:, "axis_label_curr"] = labels_df.loc[labels_df["axis_label_curr"] != 0, "axis_label_curr"].to_numpy()
            mlp_df.loc[:, "fin_label_pd"] = labels_df.loc[
                labels_df["axis_label_curr"] != 0, "fin_label_pd"].to_numpy()

            # update training DF
            if (mlp_df.shape[0] > 10) & (train_counter > 2):

                _, Y_pd, _ = fit_mlp(None, mdl, mlp_df)

                labels_df["axis_label_pd"] = Y_pd
                labels_u = [1, 2, 3]
                pd_mask = pd_layer_axis.data
                for lb in labels_u:
                    pd_ids = labels_df.loc[labels_df["axis_label_pd"] == lb, "nucleus_id"].values
                    pd_mask[np.isin(mask_zarr, pd_ids)] = lb

                pd_layer_axis.data = pd_mask

                train_counter = 0

            else:
                train_counter += 1

        else:
            pass

        lb_data = lb_layer.data
        lb_data[mask_zarr == 0] = 0
        lb_layer.data = lb_data

def curate_pec_fins(root, experiment_date, well_num, seg_model, seg_type, time_int=0, fluo_flag=False, mlp_arch=None,
                         use_model_priors=True, intra_well_only=True):

    if mlp_arch is None:
        mlp_arch = (256, 64)

    # initialize global variables
    global mlp_df, mdl, point_df, labels_df, train_counter, Y_probs, mask_zarr, label_mask, pd_layer_axis, lb_layer, seg_type_global
    seg_type_global = seg_type

    train_counter = 0

    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"

    prob_zarr, mask_zarr, scale_vec = load_zarr_data(root, seg_model, experiment_date, file_prefix, time_int)

    # load point features and labels
    point_df, labels_df, point_prefix, point_path_out = load_points_and_labels(root, file_prefix, time_int, fluo_flag)

    # check for previously-trained models and curation data
    mlp_df_all, mdl = load_mlp_data_v2(root=root, mlp_arch=mlp_arch, intra_well_only=intra_well_only)

    mlp_df_refined = update_mlp_data(labels_df, mlp_df_all, point_df, intra_well_only)

    # get frame-specific labeled points
    exp_filter = mlp_df_refined["experiment_date"].astype(str) == experiment_date
    time_filter = mlp_df_refined["time_int"] == time_int
    well_filter = mlp_df_refined["well_num"] == well_num
    mlp_df = mlp_df_refined.loc[exp_filter & time_filter & well_filter]

    # perform initial fit if we have enough local or cross-well training data
    if False: #(len(mlp_df_all) > 10) and (not fluo_flag):
        labels_df, _, _ = fit_mlp(labels_df, mdl, mlp_df_all)
        if use_ref_points == True:
            mlp_df, labels_df = sample_reference_points(mlp_df, labels_df, point_df, npoints=50)

    elif (len(mlp_df_refined) > 10):
        print("Loading prior manual labels")
        labels_df, _, _ = fit_mlp(labels_df, mdl, mlp_df_refined)
        mlp_df, labels_df = sample_reference_points(mlp_df, labels_df, point_df, npoints=250)
    else:
        labels_df.loc[:, "axis_label_pd"] = np.random.choice(np.asarray([1, 2, 3]), labels_df.shape[0])

    if ("label_pd" in point_df.columns) and use_model_priors:
        print("Loading model predictions")
        labels_df.loc[:, "fin_label_pd"] = point_df.loc[:, "label_pd"] + 1
        point_df["fin_label_pd"] = labels_df.loc[:, "fin_label_pd"]
        # mlp_df, labels_df = sample_reference_points(mlp_df, labels_df, point_df, npoints=250)
    else:
        raise Exception("No tissue predictions found")
        labels_df.loc[:, "fin_label_pd"] = np.random.choice(np.asarray([1, 2, 3, 4]), labels_df.shape[0])

    # initialize viewer
    viewer = napari.Viewer()
    viewer.add_image(prob_zarr, colormap="gray", scale=scale_vec,
                               contrast_limits=(-4, np.percentile(prob_zarr, 99.8)))


    # map from points set to label masks
    labels_u = [1, 2, 3]

    pd_mask = np.zeros_like(mask_zarr)
    pd_mask_axis = np.zeros_like(mask_zarr)
    lb_mask = np.zeros_like(mask_zarr)
    for lb in labels_u:
        pd_ids_axis = labels_df.loc[labels_df["axis_label_pd"] == lb, "nucleus_id"].values
        pd_mask_axis[np.isin(mask_zarr, pd_ids_axis)] = lb
        lb_ids = labels_df.loc[labels_df["axis_label_curr"] == lb, "nucleus_id"].values
        lb_mask[np.isin(mask_zarr, lb_ids)] = lb

    for lb in [1, 2, 3, 4]:
        pd_ids = labels_df.loc[labels_df["fin_label_pd"] == lb, "nucleus_id"].values
        pd_mask[np.isin(mask_zarr, pd_ids)] = lb


    lb_layer = viewer.add_labels(lb_mask, scale=scale_vec, name='labels', opacity=1.0, visible=True)
    pd_layer_axis = viewer.add_labels(pd_mask_axis, scale=scale_vec, name='axis prediction', opacity=0.25, visible=True)
    pd_layer = viewer.add_labels(pd_mask, scale=scale_vec, name='tissue prediction (static)', opacity=0.25, visible=False)


    lb_layer.events.set_data.connect(label_update_function)
    lb_layer.events.paint.connect(label_update_function)
    lb_layer.events.data.connect(label_update_function)
    # pd_layer.events.selected_label.connect(on_label_change)
    napari.run()

    print("Saving...")

    # add latest predictions
    # labels_df["fin_label_pd"] = pd_layer.features
    # labels_df[["oultier_prob", "fin_prob", "yolk_prob", "body_prob"]] = Y_probs
    labels_df = labels_df.dropna(axis=1, how="all")

    wait = input("Press x to approve labels for training. \nOtherwise, press Enter then Enter.")
    if 'x' in wait:
        labels_df["axis_label_final"] = labels_df["axis_label_pd"] #(labels_df["fin_label_pd"]*4 - 1).astype(int)
        override_filter = (labels_df["axis_label_final"] != labels_df["axis_label_curr"]) & (labels_df["axis_label_curr"] != 0)
        labels_df.loc[override_filter, "axis_label_final"] = labels_df.loc[override_filter, "axis_label_curr"]
    else:
        labels_df["axis_label_final"] = np.nan

    # # save MLP and MLP training data
    # if len(mlp_df_all) > 0:
    #     mlp_df = pd.concat([mlp_df, mlp_df_all], axis=0, ignore_index=True)
    # mlp_df = mlp_df.drop_duplicates(keep="first", subset=["experiment_date", "well_num", "time_int", "nucleus_id"], ignore_index=True)
    #
    # # dump(mdl, mdl_path)
    # mlp_df = mlp_df.dropna(axis=1, how="all")
    # mlp_df.to_csv(mlp_data_path, index=False)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date = "20240712_02"# "20240712_01"
    overwrite = True
    fluo_flag = False
    use_model_priors = True
    seg_model = "tdTom-bright-log-v5" #"tdTom-bright-log-v5"  # "tdTom-dim-log-v3"
    # point_model = "point_models_pos"
    well_num = 19
    time_int = 0
    curate_pec_fins(root, experiment_date=experiment_date, well_num=well_num, seg_type="tissue_only_best_model_tissue", #seg_type="seg01_best_model_tbx5a", #
                    seg_model=seg_model, time_int=time_int, mlp_arch=(128, 64),
                    fluo_flag=fluo_flag, intra_well_only=True, use_model_priors=use_model_priors)



