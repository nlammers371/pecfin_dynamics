# from aicsimageio import AICSImage
import numpy as np
import napari
import os
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

def sample_reference_points(mlp_df, labels_df, point_df, curation_df, npoints=50):

    # select reference points
    ref_points, ref_indices = farthest_point_sample(labels_df.loc[:, ["Z", "Y", "X"]].to_numpy(), npoints)

    # transfer labels
    curation_df.loc[ref_indices, "fin_label_curr"] = curation_df.loc[ref_indices, "fin_label_pd"]
    labels_df.loc[ref_indices, "fin_label_curr"] = (
            curation_df.loc[ref_indices, "fin_label_curr"] * 4 - 1).astype(int)

    curation_df.loc[labels_df["fin_label_curr"] == 0, "outlier_flags"] = True
    curation_df.loc[labels_df["fin_label_curr"] == 1, "fin_flags"] = True
    curation_df.loc[labels_df["fin_label_curr"] == 2, "yolk_flags"] = True
    curation_df.loc[labels_df["fin_label_curr"] == 3, "body_flags"] = True

    # add to training set
    mlp_df_temp = point_df.loc[labels_df["fin_label_curr"] != -1]
    mlp_df_temp.loc[:, "fin_label_curr"] = labels_df.loc[labels_df["fin_label_curr"] != -1, "fin_label_curr"].copy()
    mlp_df_temp.loc[:, "well_num"] = well_num
    mlp_df_temp.loc[:, "time_int"] = time_int

    if len(mlp_df) > 0:
        mlp_df = pd.concat([mlp_df, mlp_df_temp])
        mlp_df = mlp_df.drop_duplicates(keep="first", subset=["experiment_date", "well_num", "time_int", "nucleus_id"], ignore_index=True)
    else:
        mlp_df = mlp_df_temp.copy()

    return mlp_df, labels_df, curation_df

def strip_dummy_cols(df):
    cols = df.columns
    keep_cols = [col for col in cols if "Unnamed" not in col]
    df = df[keep_cols]
    return df


def calculate_adjacency_graph(df, k_nn=5):

    # calculate KD tree and use this to determine k nearest neighbors for each point
    xyz_array = df[["X", "Y", "Z"]]

    # print(xyz_array)
    tree = KDTree(xyz_array)

    # get nn distances
    nearest_dist, nearest_ind = tree.query(xyz_array, k=k_nn + 1)

    # find average distance to kth closest neighbor
    mean_nn_dist_vec = np.mean(nearest_dist, axis=0)
    nn_thresh = mean_nn_dist_vec[k_nn]

    # iterate through points and build adjacency network
    G = nx.Graph()

    for n in range(xyz_array.shape[0]):
        node_to = str(n)
        nodes_from = [str(f) for f in nearest_ind[n, 1:]]
        weights_from = [w for w in nearest_dist[n, 1:]]

        # add node
        G.add_node(node_to)
        # only allow edges that are within twice the average k_nn distance
        allowed_edges = np.where(weights_from <= 2 * nn_thresh)[0]
        for a in allowed_edges:
            G.add_edge(node_to, nodes_from[a], weight=weights_from[a])

    return G

# calculate shortest paths to selected point in point cloud
def calculate_network_distances(point, G):

    dist_array = np.empty((len(G)))
    if point:
        graph_distances = nx.single_source_dijkstra_path_length(G, str(point))

        for d in range(len(G)):
            try:
                dist_array[d] = graph_distances[str(d)]
            except:
                dist_array[d] = -1

        max_dist = np.max(dist_array)
        dist_array[np.where(dist_array == -1)[0]] = max_dist + 1
    else:
        dist_array[:] = 100

    return dist_array

def fit_mlp(curation_df, mdl, mlp_df):
    feature_cols = []

    feature_cols += [c for c in mlp_df.columns if "feat" in c] #+ ["well_num", "time_int", "date_norm"]
    X_train = mlp_df.loc[:, feature_cols]


    Y_train = mlp_df.loc[:, "fin_label_curr"].to_numpy()

    if binary_flag_global:
        Y_train[Y_train != 1] = 0

    print("Updating tissue predictions...")
    mdl = mdl.fit(X_train, Y_train)

    # get new predictions
    X_pd = point_df.loc[:, feature_cols]
    # nn_rows = np.where(np.all(~np.isnan(X_pd), axis=1))[0]
    Y_probs = mdl.predict_proba(X_pd)
    # Y_pd_point = np.argmax(Y_probs, axis=1)

    input_classes = np.unique(Y_train)
    # Y_probs_temp = np.zeros((Y_probs.shape[0], 4))
    Y_probs_full = np.zeros((X_pd.shape[0], 4))
    in_flags = np.isin(np.arange(4), input_classes)
    # Y_probs_temp[:, in_flags] = Y_probs
    Y_probs_full[:, in_flags] = Y_probs

    Y_pd = np.argmax(Y_probs_full, axis=1)

    if curation_df is not None:
        curation_df.loc[:, "fin_label_pd"] = (Y_pd + 1) / 4
    else:
        curation_df = None

    return curation_df, Y_pd, mdl, Y_probs_full

def get_curation_data(labels_df, mlp_df, point_df, well_num, time_int):

    # generate temporary DF to keep track of curation labels
    curation_df = pd.DataFrame(labels_df.loc[:, "fin_label_curr"].copy())
    curation_df["fin_label_pd"] = 0
    # 0=outlier, 1=fin, 2=yolk, 3=body
    curation_df["outlier_flags"] = curation_df["fin_label_curr"] == 0
    curation_df["fin_flags"] = curation_df["fin_label_curr"] == 1
    curation_df["yolk_flags"] = curation_df["fin_label_curr"] == 2
    curation_df["body_flags"] = curation_df["fin_label_curr"] == 3

    curation_df.loc[:, "fin_label_curr"] += 1
    curation_df.loc[:, "fin_label_curr"] = curation_df.loc[:, "fin_label_curr"] / 4
        # for labeling convenience

    # generate wide feature DF for classifier training
    mlp_df_temp = point_df.loc[labels_df["fin_label_curr"] != -1]
    mlp_df_temp.loc[:, "fin_label_curr"] = labels_df.loc[labels_df["fin_label_curr"] != -1, "fin_label_curr"].copy()
    # if binary_flag_global:
    #     mlp_df_temp.loc[mlp_df_temp["fin_label_curr"] != 1, "fin_label_curr"] = 0
    mlp_df_temp.loc[:, "well_num"] = well_num
    mlp_df_temp.loc[:, "time_int"] = time_int

    if len(mlp_df) > 0:
        mlp_df = pd.concat([mlp_df_temp, mlp_df])
        mlp_df = mlp_df.drop_duplicates(keep="first", subset=["experiment_date", "well_num", "time_int", "nucleus_id"], ignore_index=True)
    else:
        mlp_df = mlp_df_temp.copy()

    mlp_df = strip_dummy_cols(mlp_df)
    mlp_df = mlp_df.dropna()

    return curation_df, mlp_df


def load_mlp_data(root, curation_folder, mlp_arch):

    curated_data_dir = os.path.join(root, "metadata", "fin_curation", curation_folder, "")
    if not os.path.isdir(curated_data_dir):
        os.makedirs(curated_data_dir)
    # mdl_path = os.path.join(curated_data_dir, experiment_date + "_MLP_mdl.joblib")
    mlp_data_path = os.path.join(curated_data_dir,  "MLP_data.csv")

    mdl = MLPClassifier(max_iter=5000, hidden_layer_sizes=mlp_arch)
    if os.path.isfile(mlp_data_path):
        mlp_df_temp = pd.read_csv(mlp_data_path)
        keep_cols = [col for col in mlp_df_temp.columns if "Unnamed" not in col]
        mlp_df_temp = mlp_df_temp.loc[:, keep_cols]

        # get full mlp_df
        mlp_df = mlp_df_temp.copy()

    else:
        mlp_df = []

    return mlp_df, mdl, mlp_data_path

def load_zarr_data(root, seg_model, experiment_date, file_prefix, time_int):

    # path to raw data
    raw_zarr_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, file_prefix + ".zarr")
    prob_zarr_path = os.path.join(root, "built_data", "cellpose_output", seg_model, experiment_date,
                                  file_prefix + "_probs.zarr")

    data_zarr = zarr.open(raw_zarr_path, mode="r")
    prob_zarr = zarr.open(prob_zarr_path, mode="r")

    # convert scale vec to tuple
    scale_vec = data_zarr.attrs["voxel_size_um"]
    scale_vec = tuple(scale_vec)

    # load the specific time point we want
    im_prob = prob_zarr[time_int]

    return im_prob, scale_vec


def load_points_and_labels(root, file_prefix, time_int):

    # check for point cloud dataset
    point_prefix = file_prefix + f"_time{time_int:04}"
    point_path = os.path.join(root, "point_cloud_data", "nucleus_point_features", "")
    if not binary_flag_global:
        point_path_out = os.path.join(root, "point_cloud_data", "fin_segmentation", "")
    else:
        point_path_out = os.path.join(root, "point_cloud_data", "fin_segmentation_bin", "")
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
        labels_df["fin_label_curr"] = -1

    return point_df, labels_df, point_prefix, point_path_out

def on_points_click(layer, event):
    global mlp_df, mdl, train_counter

    if event.type == 'mouse_press' and event.button == 1 and layer.mode == "select":  # Left mouse button
        selected_index = layer.get_value(event.position, world=True, view_direction=event.view_direction,
                                         dims_displayed=event.dims_displayed)

        if selected_index is not None:
            layer_features = layer.features.copy()

            # toggle
            new_val = ~layer.features.iloc[selected_index, 0]
            layer_features.iloc[selected_index, 0] = new_val

            # reassign
            layer.features = layer_features

            # update point layer features
            if not new_val:
                point_features = point_layer.features.copy()
                point_features.iloc[selected_index, 0] = 0
                point_layer.features = point_features
            elif layer.name == 'outlier points':
                point_features = point_layer.features.copy()
                point_features.iloc[selected_index, 0] = 1 / 4
                point_layer.features = point_features
            elif layer.name == 'fin points':
                point_features = point_layer.features.copy()
                point_features.iloc[selected_index, 0] = 2 / 4
                point_layer.features = point_features
            elif layer.name == 'yolk points':
                point_features = point_layer.features.copy()
                point_features.iloc[selected_index, 0] = 3 / 4
                point_layer.features = point_features
            elif layer.name == 'body points':
                point_features = point_layer.features.copy()
                point_features.iloc[selected_index, 0] = 4 / 4
                point_layer.features = point_features

            # update data frames
            labels_df.loc[:, "fin_label_curr"] = ((point_layer.features.copy() * 4) - 1).to_numpy().astype(int)

            # updated training DF
            mlp_df = point_df.loc[labels_df["fin_label_curr"] != -1]
            mlp_df.reset_index(inplace=True, drop=True)
            mlp_df.loc[:, "fin_label_curr"] = labels_df.loc[labels_df["fin_label_curr"] != -1, "fin_label_curr"].to_numpy()
            # if binary_flag_global:
            #     mlp_df.loc[mlp_df["fin_label_curr"] != 1, "fin_label_curr"] = 0
            # update training DF

            if (mlp_df.shape[0] > 10) & (train_counter > 2):

                _, Y_pd, _, Y_probs = fit_mlp(None, mdl, mlp_df)

                # Update pd layer features
                pd_features = pd_layer.features.copy()
                pd_features.iloc[:, 0] = (Y_pd + 1) / 4
                pd_layer.features = pd_features

                train_counter = 0

            else:
                train_counter += 1

        else:
            pass

        return


def curate_pec_fins(root, experiment_date, well_num, seg_model, time_int=0, binary_flag=False, mlp_arch=None,
                        curation_folder=None, use_ref_points=True):

    if curation_folder is None:
        curation_folder = experiment_date

    if binary_flag:
        curation_folder += "_binary"

    if mlp_arch is None:
        mlp_arch = (256, 64)
    # initialize global variables
    global mlp_df, mdl, point_df, labels_df, train_counter, Y_probs, binary_flag_global
    binary_flag_global = binary_flag

    train_counter = 0

    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"

    prob_zarr, scale_vec = load_zarr_data(root, seg_model, experiment_date, file_prefix, time_int)

    # load point features and labels
    point_df, labels_df, point_prefix, point_path_out = load_points_and_labels(root, file_prefix, time_int)

    # check for previously-trained models and curation data
    mlp_df_all, mdl, mlp_data_path = load_mlp_data(root, curation_folder, mlp_arch)

    # second round of global variables
    global curation_df, point_layer, global_df, pd_layer, nn_distances, nn_indices

    curation_df, mlp_df_all = get_curation_data(labels_df, mlp_df_all, point_df, well_num, time_int)
    # rescale labels
    # curation_df.loc[:, "fin_label_curr"] = curation_df.loc[:, "fin_label_curr"] / 4

    # get frame-specific labeled points
    exp_filter = mlp_df_all["experiment_date"].astype(str) == experiment_date
    time_filter = mlp_df_all["time_int"] == time_int
    well_filter = mlp_df_all["well_num"] == well_num
    mlp_df = mlp_df_all.loc[exp_filter & time_filter & well_filter]

    # perform initial fit if we have enough local or cross-well training data
    if len(mlp_df) > 10:
        curation_df, _, _, Y_probs = fit_mlp(curation_df, mdl, mlp_df)
        if use_ref_points == True:
            mlp_df, labels_df, curation_df = sample_reference_points(mlp_df, labels_df, point_df, curation_df, npoints=50)

    elif len(mlp_df_all) > 10:
        curation_df, _, _, Y_probs = fit_mlp(curation_df, mdl, mlp_df_all)
        if use_ref_points == True:
            mlp_df, labels_df, curation_df = sample_reference_points(mlp_df, labels_df, point_df, curation_df, npoints=50)

    else:
        curation_df.loc[:, "fin_label_pd"] = np.random.choice(np.asarray([0.25, 0.5, 0.75, 1.0]), curation_df.shape[0])

    # define colormaps
    lb_colormap = vispy.color.Colormap(["white", "gray", "green", "red", "blue"], interpolation='zero',
                                           controls=[0, 0.125, 0.375, .625, .875, 1.0])

    # set colormap
    label_color_cycle = ["white", "gray", "green", "red", "blue"]

    # initialize viewer
    viewer = napari.view_image(prob_zarr, colormap="gray", scale=scale_vec,
                               contrast_limits=(-4, np.percentile(prob_zarr, 99.8)))


    point_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='point labels',
                                    size=3, features=curation_df.loc[:, "fin_label_curr"], face_color="fin_label_curr",
                                    face_colormap=lb_colormap, visible=True, face_contrast_limits=[0, 1], out_of_slice_display=True)

    ## Add outlier label layer
    if curation_df.loc[0, "outlier_flags"] == 0:
        outlier_cycle = [label_color_cycle[0]] + [label_color_cycle[1]]
    else:
        outlier_cycle = [label_color_cycle[1]] + [label_color_cycle[0]]
    outlier_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='outlier points',
                                      size=4, features=curation_df.loc[:, "outlier_flags"], face_color="outlier_flags",
                                      face_color_cycle=outlier_cycle, visible=False, out_of_slice_display=True)

    ## Add fin label layer
    if curation_df.loc[0, "fin_flags"] == 0:
        fin_cycle = [label_color_cycle[0]] + [label_color_cycle[2]]
    else:
        fin_cycle = [label_color_cycle[2]] + [label_color_cycle[0]]

    fin_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='fin points',
                                  size=4, features=curation_df.loc[:, "fin_flags"],
                                  face_color="fin_flags",
                                  face_color_cycle=fin_cycle, visible=False, out_of_slice_display=True)

    ## Add yolk label layer
    if curation_df.loc[0, "yolk_flags"] == 0:
        yolk_cycle = [label_color_cycle[0]] + [label_color_cycle[3]]
    else:
        yolk_cycle = [label_color_cycle[3]] + [label_color_cycle[0]]

    yolk_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='yolk points',
                                   size=3, features=curation_df.loc[:, "yolk_flags"],
                                   face_color="yolk_flags",
                                   face_color_cycle=yolk_cycle, visible=False, out_of_slice_display=True)

    ## Add body label layer
    if curation_df.loc[0, "body_flags"] == 0:
        body_cycle = [label_color_cycle[0]] + [label_color_cycle[4]]
    else:
        body_cycle = [label_color_cycle[4]] + [label_color_cycle[0]]
    body_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='body points',
                                   size=3, features=curation_df.loc[:, "body_flags"],
                                   face_color="body_flags",
                                   face_color_cycle=body_cycle, visible=False, out_of_slice_display=True)

    if True: #not binary_flag:
        pd_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='prediction',
                                     size=3, features=curation_df.loc[:, "fin_label_pd"], opacity=0.5,
                                     face_color="fin_label_pd",
                                     # face_color_cycle=label_color_cycle, visible=True)
                                     face_colormap=lb_colormap, visible=True, face_contrast_limits=[0, 1], out_of_slice_display=True)
    # else:
    #     pd_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='prediction',
    #                                  size=4, features=curation_df.loc[:, "fin_label_pd"], opacity=0.5,
    #                                  face_color="fin_label_pd",
    #                                  # face_color_cycle=label_color_cycle, visible=True)
    #                                  face_colormap=lb_colormap, visible=True, face_contrast_limits=[0, 1],
    #                                  out_of_slice_display=True)


    # connect to event trigger function
    outlier_layer.mouse_drag_callbacks.append(on_points_click)
    fin_layer.mouse_drag_callbacks.append(on_points_click)
    yolk_layer.mouse_drag_callbacks.append(on_points_click)
    body_layer.mouse_drag_callbacks.append(on_points_click)

    napari.run()

    print("Saving...")

    # add latest predictions
    labels_df["fin_label_pd"] = pd_layer.features
    # labels_df[["oultier_prob", "fin_prob", "yolk_prob", "body_prob"]] = Y_probs
    labels_df = labels_df.dropna(axis=1, how="all")

    wait = input("Press x to approve labels for training. \nOtherwise, press Enter then Enter.")
    labels_df["binary_flag"] = binary_flag
    if 'x' in wait:
        labels_df["fin_label_final"] = (labels_df["fin_label_pd"]*4 - 1).astype(int)
        if not binary_flag:
            override_filter = (labels_df["fin_label_final"] != labels_df["fin_label_curr"]) & (labels_df["fin_label_curr"] != -1)
            labels_df.loc[override_filter, "fin_label_final"] = labels_df.loc[override_filter, "fin_label_curr"]
        else:
            manual_labels = labels_df["fin_label_curr"].to_numpy()
            manual_labels[manual_labels != 1] = 0
            override_filter = (labels_df["fin_label_final"] != manual_labels) & (
                        labels_df["fin_label_curr"] != -1)
            labels_df.loc[override_filter, "fin_label_final"] = manual_labels[override_filter]
    else:
        labels_df["fin_label_final"] = np.nan

    # save condensed version without the features
    labels_df.to_csv((point_path_out + point_prefix + "_labels.csv"), index=False)

    # save MLP and MLP training data
    mlp_df = pd.concat([mlp_df, mlp_df_all], axis=0, ignore_index=True)
    mlp_df = mlp_df.drop_duplicates(keep="first", subset=["experiment_date", "well_num", "time_int", "nucleus_id"], ignore_index=True)

    # dump(mdl, mdl_path)
    mlp_df = mlp_df.dropna(axis=1, how="all")
    mlp_df.to_csv(mlp_data_path, index=False)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date = "20240619"
    overwrite = True
    binary_flag = True
    seg_model = "tdTom-bright-log-v5"
    # point_model = "point_models_pos"
    well_num = 3
    curation_folder = experiment_date
    time_int = 90
    curate_pec_fins(root, curation_folder=curation_folder, experiment_date=experiment_date, well_num=well_num,
                    seg_model=seg_model, time_int=time_int, mlp_arch=(256, 64), binary_flag=binary_flag)

