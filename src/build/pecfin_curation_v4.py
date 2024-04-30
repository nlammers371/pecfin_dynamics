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
from skimage.measure import regionprops



def on_points_click(layer, event):
    global mlp_df, mdl

    if event.type == 'mouse_press' and event.button == 1 and layer.mode == "select":  # Left mouse button
        # selected_index = layer.get_value(event.position, world=True)
        # print(f"Selected point index: {selected_index}")
        selected_index = layer.get_value(event.position, world=True, view_direction=event.view_direction,
                                         dims_displayed=event.dims_displayed)
        if selected_index is not None:
            layer_features = layer.features.copy()
            new_val = ~layer.features.iloc[selected_index, 0]
            layer_features.iloc[selected_index, 0] = new_val
            layer.features = layer_features

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
            start = time.time()
            point_df.loc[:, "fin_label_curr"] = ((point_layer.features.copy() * 4) - 1).to_numpy().astype(int)

            mlp_df_temp = point_df.loc[point_df["fin_label_curr"] != -1]
            # mlp_df_temp.loc[:, "well_num"] = well_num
            # mlp_df_temp.loc[:, "time_int"] = time_int
            # mlp_df_temp.loc[:, global_df.columns.tolist()] = np.tile(global_df.iloc[:, :].to_numpy(),
            #                                                            (mlp_df_temp.shape[0], 1)).copy()

            train_orig = mlp_df.copy()
            mlp_df = pd.concat([train_orig, mlp_df_temp], axis=0)
            mlp_df = mlp_df[~mlp_df.index.duplicated(keep='first')]

            # fit the classifier
            feature_cols = [c for c in mlp_df.columns if "feat" in c]
            X_train = mlp_df.loc[:, feature_cols]
            Y_train = mlp_df.loc[:, "fin_label_curr"]

            if X_train.shape[0] > 10:
                print("Updating tissue predictions...")
                mdl = mdl.fit(X_train, Y_train)

                # get new predictions
                X_pd = point_df.loc[:, feature_cols]
                Y_pd = mdl.predict(X_pd)
                print(time.time() - start)

                # Update pd layer features
                # pd_features = pd_layer.features.copy()
                # pd_features = Y_pd + 1
                pd_features = pd_layer.features.copy()
                pd_features.iloc[:, 0] = (Y_pd + 1) / 4
                pd_layer.features = pd_features

            # print(pd_layer.features.head(4))
            # pd_layer.loc[:, "fin_label_pd"] = Y_pd.copy()
        else:
            pass

        return


def curate_pec_fins(root, experiment_date, scale_vec, seg_model, well_num, time_int=0,
                    overwrite_flag=False, n_mlp_nodes=10):
    
    global mlp_df, mdl, point_df

    # load metadata
    # metadata_path = os.path.join(root, "metadata", experiment_date + "_master_metadata_df.csv")
    # metadata_df = pd.read_csv(metadata_path, index_col=0)

    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"

    # load segmentation data
    mask_zarr_path = os.path.join(root, "built_data", "stitched_labels", seg_model, experiment_date,
                                 file_prefix + "_labels_stitched.zarr")

    prob_zarr_path = os.path.join(root, "built_data", "cellpose_output", seg_model, experiment_date, file_prefix + "_probs.zarr")

    # mask_zarr = zarr.open(mask_zarr_path, mode="r")
    prob_zarr = zarr.open(prob_zarr_path, mode="r")
    
    # load the specific time point we want
    im_prob = prob_zarr[time_int]
    # im_mask = mask_zarr[time_int]
    
    # check for point cloud dataset
    point_prefix = file_prefix + f"_time{time_int:04}"
    point_path = os.path.join(root, "built_data", "processed_point_clouds", experiment_date, "")
    point_path_out = os.path.join(root, "built_data", "point_clouds", "feature_predictions", experiment_date, "")
    if not os.path.isdir(point_path_out):
        os.makedirs(point_path_out)
    point_df = pd.read_csv(point_path + point_prefix + "_centroids.csv")

    # check for previously-trained models and curation data
    mdl_dir = os.path.join(root, "metadata", "fin_curation", "")
    if not os.path.isdir(mdl_dir):
        os.makedirs(mdl_dir)
    mdl_path = os.path.join(mdl_dir, experiment_date + "_MLP_mdl.joblib")
    curated_data_dir = os.path.join(root, "metadata", "fin_curation", "")
    if not os.path.isdir(curated_data_dir):
        os.makedirs(curated_data_dir)
    curated_data_path = os.path.join(curated_data_dir, experiment_date + "_MLP_data.csv")
    # if os.path.isfile(mdl_path):
    #     mdl = load(mdl_path)
    # else:
    mdl = MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=n_mlp_nodes)
    if os.path.isfile(curated_data_path):
        mlp_df = pd.read_csv(curated_data_path)
        keep_cols = [col for col in mlp_df.columns if "Unnamed" not in col]
        mlp_df = mlp_df.loc[:, keep_cols]
    else:
        mlp_df = []

    # add column to track fin segmentation
    if "fin_curation_flag" not in point_df.columns:
        point_df["fin_curation_flag"] = False
        point_df["fin_curation_date"] = np.nan

    # make directory for saving fin masks
    fin_point_dir = os.path.join(root, "built_data", "processed_point_clouds", experiment_date, "")
    
    # convert scale vec to tuple
    scale_vec = tuple(scale_vec)

    # define colormaps
    lb_colormap = vispy.color.Colormap(["white", "gray", "green", "red", "blue"], interpolation='zero',
                                       controls=[0, 0.125, 0.375, .625, .875, 1.0])

    global curation_df, point_layer, global_df, pd_layer


    fname_suffix = f"well{well_num:03}_t{time_int:03}"

    # load point cloud
    # point_cloud_options = glob(os.path.join(fin_point_dir, "*" + fname_suffix + "*"))
    #     # local_options = [p for p in point_cloud_options if "global" not in p]
    #     # global_options = [p for p in point_cloud_options if "global" in p]
    #     # if (len(local_options) > 1) | (len(global_options) > 1):
    #     #     raise Exception("Found multiple files with same well and time index.")
    #     # point_df = pd.read_csv(local_options[0], index_col=0)
    #     # global_df = pd.read_csv(global_options[0], index_col=0)
    #     # point_df.loc[:, global_df.columns.tolist()] = np.tile(global_df.iloc[:, :].to_numpy(),
    #     #                                                       (point_df.shape[0], 1)).copy()
    
    if "fin_label_curr" not in point_df.columns:
        point_df["fin_label_curr"] = -1

    # generate temporary DF to keep track of curation labels
    curation_df = pd.DataFrame(point_df.loc[:, "fin_label_curr"].copy())
    curation_df["fin_label_pd"] = 0
    # 0=outlier, 1=fin, 2=yolk, 3=body
    curation_df["outlier_flags"] = curation_df["fin_label_curr"] == 0
    curation_df["fin_flags"] = curation_df["fin_label_curr"] == 1
    curation_df["yolk_flags"] = curation_df["fin_label_curr"] == 2
    curation_df["body_flags"] = curation_df["fin_label_curr"] == 3

    curation_df.loc[:, "fin_label_curr"] += 1  # for labeling convenience

    # generate wide feature DF for classifier training
    mlp_df_temp = point_df.loc[point_df["fin_label_curr"] != -1]
    mlp_df_temp.loc[:, "well_num"] = well_num
    mlp_df_temp.loc[:, "time_int"] = time_int
    # mlp_df_temp.loc[:, global_df.columns.tolist()] = np.tile(global_df.iloc[:, :].to_numpy(), (mlp_df_temp.shape[0], 1)).copy()
    if len(mlp_df) > 0:
        mlp_df = pd.concat([mlp_df, mlp_df_temp])
        mlp_df = mlp_df.drop_duplicates(ignore_index=True)
    else:
        mlp_df = mlp_df_temp.copy()

    curation_df.loc[:, "fin_label_curr"] = curation_df.loc[:, "fin_label_curr"] / 4
    if len(mlp_df) > 10:
        feature_cols = [c for c in mlp_df.columns if "feat" in c]
        X_train = mlp_df.loc[:, feature_cols]
        Y_train = mlp_df.loc[:, "fin_label_curr"]

        print("Updating tissue predictions...")
        mdl = mdl.fit(X_train, Y_train)

        # get new predictions
        X_pd = point_df.loc[:, feature_cols]
        Y_pd = mdl.predict(X_pd)
        curation_df.loc[:, "fin_label_pd"] = (Y_pd + 1) / 4
    else:
        curation_df.loc[:, "fin_label_pd"] = curation_df.loc[:, "fin_label_curr"].copy()

    # set colormap
    label_color_cycle = ["white", "gray", "green", "red", "blue"]

    # initialize viewer
    viewer = napari.view_image(im_prob, colormap="gray", scale=scale_vec,
                               contrast_limits=(-16, np.percentile(im_prob, 99.8)))


    # generate master point array to integrate results
    point_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='point labels',
                                    size=4, features=curation_df.loc[:, "fin_label_curr"], face_color="fin_label_curr",
                                    # face_color_cycle=label_color_cycle, visible=True)
                                    face_colormap=lb_colormap, visible=True)

    outlier_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='outlier points',
                                      size=4, features=curation_df.loc[:, "outlier_flags"], face_color="outlier_flags",
                                      face_color_cycle=label_color_cycle[:2], visible=False)

    if curation_df.loc[0, "fin_flags"] == 0:
        fin_cycle = [label_color_cycle[0]] + [label_color_cycle[2]]
    else:
        fin_cycle = [label_color_cycle[2]] + [label_color_cycle[0]]

    fin_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='fin points',
                                  size=4, features=curation_df.loc[:, "fin_flags"],
                                  face_color="fin_flags",
                                  face_color_cycle=fin_cycle, visible=False)

    yolk_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='yolk points',
                                   size=4, features=curation_df.loc[:, "yolk_flags"],
                                   face_color="yolk_flags",
                                   face_color_cycle=[label_color_cycle[0]] + [label_color_cycle[3]], visible=False)

    body_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='body points',
                                   size=4, features=curation_df.loc[:, "body_flags"],
                                   face_color="body_flags",
                                   face_color_cycle=[label_color_cycle[0]] + [label_color_cycle[4]], visible=False)

    pd_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='prediction',
                                 size=4, features=curation_df.loc[:, "fin_label_pd"],
                                 face_color="fin_label_pd",
                                 # face_color_cycle=label_color_cycle, visible=True)
                                 face_colormap=lb_colormap, visible=True)

    # pd_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='predictions',
    #                                 size=4, features=curation_df.loc[:, "fin_label_pd"],
    #                                 face_color="fin_label_pd", opacity=0.7, visible=True)
    #


    # connect to event trigger function
    outlier_layer.mouse_drag_callbacks.append(on_points_click)
    fin_layer.mouse_drag_callbacks.append(on_points_click)
    yolk_layer.mouse_drag_callbacks.append(on_points_click)
    body_layer.mouse_drag_callbacks.append(on_points_click)

    # # initialize points layer
    # if os.path.isfile(point_path):
    #     point_array = np.load(point_path)
    #     point_array = np.divide(point_array, scale_vec)
    #     # points_layer = viewer.add_points(point_array, name="fin hull points", size=8, scale=scale_vec,
    #     #                                  n_dimensional=True)
    #     if point_array.shape[0] >= 6:
    #         calculate_fin_hull(point_array)
    # elif carry_flag:
    #     point_array = point_array_prev
    # else:
    #     point_array = np.empty((0, 3))
    # points_layer = viewer.add_points(point_array, name="fin hull points", size=8, scale=scale_vec, n_dimensional=True)
    #
    # points_layer.events.data.connect(event_trigger)
    # point_df.loc[:, "fin_label_curr"] = point_layer.features.copy() - 1

    napari.run()

    print("Saving...")

    # add latest predictions
    point_df["fin_label_pd"] = pd_layer.features

    # save full dataset
    point_df.to_csv((point_path + point_prefix + "_centroids.csv"), index=False)

    # save condensed version without the features
    keep_cols = [col for col in point_df.columns if "feat" not in col]
    point_df_clean = point_df.loc[:, keep_cols]
    point_df_clean.to_csv((point_path_out + point_prefix + "_centroids.csv"), index=False)

    # save MLP and MLP training data
    dump(mdl, mdl_path)
    mlp_df.to_csv(curated_data_path, index=False)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\"
    experiment_date = "20240223"
    overwrite = True
    scale_vec = np.asarray([2.0, 0.55, 0.55])
    seg_model = "log-v5"
    well_num = 12
    curate_pec_fins(root, experiment_date, scale_vec, seg_model, well_num, time_int=50,
                    overwrite_flag=False, n_mlp_nodes=500)

