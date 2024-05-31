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

def strip_dummy_cols(df):
    cols = df.columns
    keep_cols = [col for col in cols if "Unnamed" not in col]
    df = df[keep_cols]
    return df

def calculate_nn_labels(labels_array, G):

    # check for user-assigned labels
    user_labels = labels_array[labels_array != 0]
    label_indices = np.where(labels_array != 0)[0]

    if len(label_indices) == 0:
        pass
    else:
        dist_array = np.ones((curation_df.shape[0], len(user_labels)))*np.inf
        for i, lb in enumerate(label_indices):
            dist_array[:, i] = calculate_network_distances(lb, G)

        nn_idx = np.argmin(dist_array, axis=1)
        labels_array = user_labels[nn_idx]

    return labels_array

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
        allowed_edges = np.where(weights_from <= (2 * nn_thresh))[0]
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

def fit_mlp(curation_df, mdl, mlp_df, fluo_col=None):
    feature_cols = []
    # if fluo_col is not None:
    #     feature_cols = [fluo_col + "_mean_nn", fluo_col + "_mean_nn_norm", fluo_col + "_dist", fluo_col + "_dist_nn"]

    feature_cols += [c for c in mlp_df.columns if "feat" in c] #+ ["well_num", "time_int", "date_norm"]
    X_train = mlp_df.loc[:, feature_cols]

    Y_train = mlp_df.loc[:, "fin_label_curr"]

    print("Updating tissue predictions...")
    mdl = mdl.fit(X_train, Y_train)

    # get new predictions
    X_pd = point_df.loc[:, feature_cols]
    Y_probs = mdl.predict_proba(X_pd)
    # Y_pd_point = np.argmax(Y_probs, axis=1)

    input_classes = np.unique(Y_train)
    Y_probs_full = np.zeros((Y_probs.shape[0], 4))
    in_flags = np.isin(np.arange(4), input_classes)
    Y_probs_full[:, in_flags] = Y_probs
    # factor in nearest neighbor stats
    nn_probs_full = Y_probs_full[nn_indices, :]
    dist_mask = np.tile(nn_distances[:, :, np.newaxis] <= 15, (1, 1, 4))
    nn_probs = np.squeeze(np.mean(np.multiply(dist_mask, nn_probs_full), axis=1))
    nn_probs = np.divide(nn_probs, np.sum(nn_probs,1)[:, np.newaxis])
    Y_pd_nn = np.argmax(nn_probs, axis=1)

    if curation_df is not None:
        curation_df.loc[:, "fin_label_pd"] = (Y_pd_nn + 1) / 4
    else:
        curation_df = None

    return curation_df, Y_pd_nn, mdl, nn_probs

def get_curation_data(labels_df, mlp_df, point_df, well_num, time_int):

    # generate temporary DF to keep track of curation labels
    curation_df = pd.DataFrame(labels_df.loc[:, "fin_label_curr"].copy())
    curation_df["fin_label_pd"] = 0
    curation_df["fin_label_nn"] = 0

    # 0=outlier, 1=fin, 2=yolk, 3=body
    curation_df["outlier_flags"] = curation_df["fin_label_curr"] == 0
    curation_df["fin_flags"] = curation_df["fin_label_curr"] == 1
    curation_df["yolk_flags"] = curation_df["fin_label_curr"] == 2
    curation_df["body_flags"] = curation_df["fin_label_curr"] == 3

    curation_df.loc[:, "fin_label_curr"] += 1  # for labeling convenience

    # generate wide feature DF for classifier training
    mlp_df_temp = point_df.loc[labels_df["fin_label_curr"] != -1]
    mlp_df_temp.loc[:, "fin_label_curr"] = labels_df.loc[labels_df["fin_label_curr"] != -1, "fin_label_curr"].copy()
    mlp_df_temp.loc[:, "well_num"] = well_num
    mlp_df_temp.loc[:, "time_int"] = time_int

    if len(mlp_df) > 0:
        mlp_df = pd.concat([mlp_df, mlp_df_temp])
        mlp_df = mlp_df.drop_duplicates(ignore_index=True)
    else:
        mlp_df = mlp_df_temp.copy()

    mlp_df = strip_dummy_cols(mlp_df)

    return curation_df, mlp_df


def load_mlp_data(root, curation_folder, n_mlp_nodes, n_layers):

    curated_data_dir = os.path.join(root, "metadata", "fin_curation", curation_folder, "")
    if not os.path.isdir(curated_data_dir):
        os.makedirs(curated_data_dir)
    # mdl_path = os.path.join(curated_data_dir, experiment_date + "_MLP_mdl.joblib")
    mlp_data_path = os.path.join(curated_data_dir,  "MLP_data.csv")

    mdl = MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=(n_mlp_nodes,) * n_layers)
    if os.path.isfile(mlp_data_path):
        mlp_df = pd.read_csv(mlp_data_path)
        keep_cols = [col for col in mlp_df.columns if "Unnamed" not in col]
        mlp_df = mlp_df.loc[:, keep_cols]
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
    try:
        scale_vec = data_zarr.attrs["voxel_size_um"]
    except:
        scale_vec = np.asarray([2.0, 0.55, 0.55])
    scale_vec = tuple(scale_vec)

    # load the specific time point we want
    im_prob = prob_zarr[time_int]

    return im_prob, scale_vec


def load_points_and_labels(root, seg_model, experiment_date, file_prefix, time_int):
    
    # check for point cloud dataset
    point_prefix = file_prefix + f"_time{time_int:04}"
    point_path = os.path.join(root, "built_data", "nucleus_data", "processed_point_clouds",  experiment_date, "")
    point_path_out = os.path.join(root, "built_data", "nucleus_data", "fin_segmentation",  experiment_date, "")

    if not os.path.isdir(point_path_out):
        os.makedirs(point_path_out)
    point_df = pd.read_csv(point_path + point_prefix + "_nuclei.csv")
    point_df = strip_dummy_cols(point_df)

    # check for pre-existing labels DF
    if os.path.isfile(point_path_out + point_prefix + "_nuclei.csv"):
        labels_df = pd.read_csv(point_path_out + point_prefix + "_nuclei.csv")
        labels_df = strip_dummy_cols(labels_df)

    else:
        keep_cols = [col for col in point_df.columns if "feat" not in col]
        labels_df = point_df.loc[:, keep_cols]
        labels_df["fin_curation_flag"] = False
        labels_df["fin_curation_date"] = np.nan
        labels_df["fin_label_curr"] = -1
        
    return point_df, labels_df, point_prefix, point_path_out

def on_points_click(layer, event):
    global mlp_df, mdl, train_counter, labels_df

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
            # start = time.time()
            labels_df.loc[:, "fin_label_curr"] = ((point_layer.features.copy() * 4) - 1).to_numpy().astype(int)

            # update nn-based assignments
            nn_features = calculate_nn_labels(point_layer.features.copy().to_numpy(), G)
            pd_features = pd_layer.features.copy()
            pd_features.iloc[:, 0] = nn_features
            pd_layer.features = pd_features

            # updated training DF
            mlp_df_temp = point_df.loc[labels_df["fin_label_curr"] != -1]
            mlp_df_temp.loc[:, "fin_label_curr"] = labels_df.loc[labels_df["fin_label_curr"] != -1, "fin_label_curr"].copy()

            # combine old and new training DFs
            train_orig = mlp_df.copy()
            mlp_df = pd.concat([train_orig, mlp_df_temp], axis=0)
            mlp_df = mlp_df[~mlp_df.index.duplicated(keep='first')]

            if (mlp_df.shape[0] > 10) & (train_counter > 15):

                _, Y_pd, _, Y_probs = fit_mlp(None, mdl, mlp_df, fluo_col)

                # Update pd layer features

                # pd_features = pd_layer.features.copy()
                # pd_features.iloc[:, 0] = (Y_pd + 1) / 4
                # pd_layer.features = pd_features

                train_counter = 0

            else:
                train_counter += 1

            # print(pd_layer.features.head(4))
            # pd_layer.loc[:, "fin_label_pd"] = Y_pd.copy()
        else:
            pass

        return


def curate_pec_fins(root, curation_folder, experiment_date, seg_model, well_num, time_int=0, fluo_label=None,
                    overwrite_flag=False, n_layers=1, n_mlp_nodes=10):
    
    # initialize global variables
    global mlp_df, mdl, point_df, fluo_col, labels_df, train_counter, Y_probs

    fluo_col = fluo_label
    train_counter = 0

    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"
    prob_zarr, scale_vec = load_zarr_data(root, seg_model, experiment_date, file_prefix, time_int)
    
    # load point features and labels
    point_df, labels_df, point_prefix, point_path_out = load_points_and_labels(root, seg_model, experiment_date,
                                                                               file_prefix, time_int)

    # check for previously-trained models and curation data
    mlp_df, mdl, mlp_data_path = load_mlp_data(root, curation_folder, n_mlp_nodes, n_layers)

    # second round of global variables
    global curation_df, point_layer, global_df, pd_layer, nn_distances, nn_indices, G

    # make NN graph for label smoothing
    nn_k = 5
    tree = KDTree(point_df[["Z", "Y", "X"]], leaf_size=2)
    nn_distances, nn_indices = tree.query(point_df[["Z", "Y", "X"]], k=nn_k + 1)

    # if fluo_label is not None:
    #     # generate adjacency graph
    #     G = calculate_adjacency_graph(point_df)
    #
    #     max_ind = np.argmax(point_df[fluo_col + '_mean_nn_norm'])
    #     fluo_distances = calculate_network_distances(max_ind, G)
    #     fluo99 = np.percentile(fluo_distances, 99.5)
    #     fluo_distances[fluo_distances > fluo99] = fluo99
    #     point_df[fluo_col + '_dist_nn'] = fluo_distances.copy()

    # process dataframes
    curation_df, mlp_df = get_curation_data(labels_df, mlp_df, point_df, well_num, time_int)

    # convert labels to fractional form
    curation_df.loc[:, "fin_label_curr"] = curation_df.loc[:, "fin_label_curr"] / 4

    # get NN label assignments
    G = calculate_adjacency_graph(point_df)
    curation_df["fin_label_nn"] = calculate_nn_labels(curation_df["fin_label_nn"].to_numpy(), G)

    # fit MLP if there are enough points
    if len(mlp_df) > 10:
        curation_df, _, _, Y_probs = fit_mlp(curation_df, mdl, mlp_df, fluo_col)
    else:
        curation_df.loc[:, "fin_label_pd"] = curation_df.loc[:, "fin_label_curr"].copy()

    # define colormaps
    lb_colormap = vispy.color.Colormap(["white", "gray", "green", "red", "blue"], interpolation='zero',
                                           controls=[0, 0.125, 0.375, .625, .875, 1.0])

    # set colormap
    label_color_cycle = ["white", "gray", "green", "red", "blue"]

    # initialize viewer
    viewer = napari.view_image(prob_zarr, colormap="gray", scale=scale_vec,
                               contrast_limits=(-4, np.percentile(prob_zarr, 99.8)))

    # generate master point array to integrate results
    if fluo_col is not None:
        marker_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='tbx5a',
                                        size=4, features=point_df.loc[:, fluo_col + "_mean_nn"],
                                        face_color=fluo_col + "_mean_nn",
                                        face_colormap="Greens", visible=True)

    point_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='point labels',
                                    size=4, features=curation_df.loc[:, "fin_label_curr"], face_color="fin_label_curr",
                                    face_colormap=lb_colormap, visible=True, face_contrast_limits=[0, 1])

    ## Add outlier label layer
    if curation_df.loc[0, "outlier_flags"] == 0:
        outlier_cycle = [label_color_cycle[0]] + [label_color_cycle[1]]
    else:
        outlier_cycle = [label_color_cycle[1]] + [label_color_cycle[0]]
    outlier_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='outlier points',
                                      size=4, features=curation_df.loc[:, "outlier_flags"], face_color="outlier_flags",
                                      face_color_cycle=outlier_cycle, visible=False)

    ## Add fin label layer
    if curation_df.loc[0, "fin_flags"] == 0:
        fin_cycle = [label_color_cycle[0]] + [label_color_cycle[2]]
    else:
        fin_cycle = [label_color_cycle[2]] + [label_color_cycle[0]]

    fin_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='fin points',
                                  size=4, features=curation_df.loc[:, "fin_flags"],
                                  face_color="fin_flags",
                                  face_color_cycle=fin_cycle, visible=False)

    ## Add yolk label layer
    if curation_df.loc[0, "yolk_flags"] == 0:
        yolk_cycle = [label_color_cycle[0]] + [label_color_cycle[3]]
    else:
        yolk_cycle = [label_color_cycle[3]] + [label_color_cycle[0]]

    yolk_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='yolk points',
                                   size=4, features=curation_df.loc[:, "yolk_flags"],
                                   face_color="yolk_flags",
                                   face_color_cycle=yolk_cycle, visible=False)

    ## Add body label layer
    if curation_df.loc[0, "body_flags"] == 0:
        body_cycle = [label_color_cycle[0]] + [label_color_cycle[4]]
    else:
        body_cycle = [label_color_cycle[4]] + [label_color_cycle[0]]
    body_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='body points',
                                   size=4, features=curation_df.loc[:, "body_flags"],
                                   face_color="body_flags",
                                   face_color_cycle=body_cycle, visible=False)

    pd_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='prediction',
                                 size=4, features=curation_df.loc[:, "fin_label_nn"].astype(np.float64), opacity=0.5,
                                 face_color="fin_label_nn",
                                 # face_color_cycle=label_color_cycle, visible=True)
                                 face_colormap=lb_colormap, visible=True, face_contrast_limits=[0, 1])


    # connect to event trigger function
    outlier_layer.mouse_drag_callbacks.append(on_points_click)
    fin_layer.mouse_drag_callbacks.append(on_points_click)
    yolk_layer.mouse_drag_callbacks.append(on_points_click)
    body_layer.mouse_drag_callbacks.append(on_points_click)

    napari.run()

    print("Saving...")

    # add latest predictions
    labels_df["fin_label_nn"] = pd_layer.features
    labels_df[["oultier_prob", "fin_prob", "yolk_prob", "body_prob"]] = Y_probs
    labels_df = labels_df.dropna(axis=1, how="all")

    wait = input("Press x to approve labels for training. \nOtherwise, press Enter then Enter.")
    if wait == 'x':
        labels_df["fin_label_final"] = labels_df["fin_label_nn"]
    else:
        labels_df["fin_label_final"] = np.nan

    # save condensed version without the features
    labels_df.to_csv((point_path_out + point_prefix + "_nuclei.csv"), index=False)

    # save MLP and MLP training data
    mlp_df = mlp_df.dropna(axis=1, how="all")
    mlp_df.to_csv(mlp_data_path, index=False)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date = "20240223"
    overwrite = True
    seg_model = "log-v5"
    curation_folder = "tbx5a_training"
    well_num = 12
    curate_pec_fins(root, curation_folder=curation_folder, experiment_date=experiment_date, seg_model=seg_model, well_num=well_num,
                    time_int=150, overwrite_flag=False, n_mlp_nodes=100, n_layers=2)

