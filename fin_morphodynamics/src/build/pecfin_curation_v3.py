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


def on_points_click(layer, event):
    global train_df, mdl

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
                point_features.iloc[selected_index, 0] = 1/4
                point_layer.features = point_features
            elif layer.name == 'fin points':
                point_features = point_layer.features.copy()
                point_features.iloc[selected_index, 0] = 2/4
                point_layer.features = point_features
            elif layer.name == 'yolk points':
                point_features = point_layer.features.copy()
                point_features.iloc[selected_index, 0] = 3/4
                point_layer.features = point_features
            elif layer.name == 'body points':
                point_features = point_layer.features.copy()
                point_features.iloc[selected_index, 0] = 4/4
                point_layer.features = point_features


            # update data frames
            start = time.time()
            point_df.loc[:, "fin_label_cur"] = ((point_layer.features.copy()*4) - 1).to_numpy().astype(int)

            train_df_temp = point_df.loc[point_df["fin_label_cur"] != -1]
            train_df_temp.loc[:, "well_index"] = well_index
            train_df_temp.loc[:, "time_index"] = time_index
            # train_df_temp.loc[:, global_df.columns.tolist()] = np.tile(global_df.iloc[:, :].to_numpy(),
            #                                                            (train_df_temp.shape[0], 1)).copy()

            train_orig = train_df.copy()
            train_df = pd.concat([train_orig, train_df_temp], axis=0)
            train_df = train_df[~train_df.index.duplicated(keep='first')]

            # fit the classifier
            feature_cols = [c for c in train_df.columns if "feat" in c]
            X_train = train_df.loc[:, feature_cols]
            Y_train = train_df.loc[:, "fin_label_cur"]
            mdl = mdl.fit(X_train, Y_train)

            # get new predictions
            X_pd = point_df.loc[:, feature_cols]
            Y_pd = mdl.predict(X_pd)
            print(time.time() - start)

            # Update pd layer features
            # pd_features = pd_layer.features.copy()
            # pd_features = Y_pd + 1
            pd_features = pd_layer.features.copy()
            pd_features.iloc[:, 0] = (Y_pd + 1)/4
            pd_layer.features = pd_features

            # print(pd_layer.features.head(4))
            # pd_layer.loc[:, "fin_label_pd"] = Y_pd.copy()
        else:
            pass


        return


def curate_pec_fins(root, experiment_date, overwrite_flag=False, n_mlp_nodes=500):
    global train_df, mdl

    # load metadata
    metadata_path = os.path.join(root, "metadata", experiment_date + "_master_metadata_df.csv")
    metadata_df = pd.read_csv(metadata_path, index_col=0)

    # check for previously-trained models and curation data
    mdl_path = os.path.join(root, "metadata",  experiment_date + "_MLP_mdl.joblib")
    curated_date_path = os.path.join(root, "metadata", experiment_date + "_MLP_data.csv")
    if os.path.isfile(mdl_path):
        mdl = load(mdl_path)
    else:
        mdl = MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=n_mlp_nodes)
    if os.path.isfile(curated_date_path):
        train_df = pd.read_csv(curated_date_path, index_col=0)
    else:
        train_df = []

    # add column to track fin segmentation
    if "fin_curation_flag" not in metadata_df.columns:
        metadata_df["fin_curation_flag"] = False
    if "fin_curation_date" not in metadata_df:
        metadata_df["fin_curation_date"] = np.nan

    
    # git list of prob files produced by CellPose
    prob_file_dir = os.path.join(root, "built_data", "cellpose_output", experiment_date, "")
    
    # make directory for saving fin masks
    fin_point_dir = os.path.join(root, "built_data", "processed_point_clouds", experiment_date, "")
    
    # Extract pixel sizes and bit_depth
    pixel_res_vec = metadata_df.loc[0, ["z_res_um", "y_res_um", "x_res_um"]].to_numpy().astype(np.float32)
    scale_vec = tuple(pixel_res_vec)
    # scale_array = np.asarray(scale_vec)
    
    # get list of fins to segment
    if not overwrite_flag:
        seg_flags = metadata_df["fin_curation_flag"] == False
        well_id_list = metadata_df.loc[seg_flags, "well_index"]
        time_id_list = metadata_df.loc[seg_flags, "time_index"]
    else:
        well_id_list = metadata_df.loc[:, "well_index"]
        time_id_list = metadata_df.loc[:, "time_index"]

    # define colormaps
    lb_colormap = vispy.color.Colormap(["white", "gray", "green", "red", "blue"], interpolation='zero',
                                       controls=[0, 0.125, 0.375, .625, .875, 1.0])


    #
    image_i = 0

    while image_i < len(well_id_list):

        global curation_df, point_layer, point_df, global_df, time_index, well_index, pd_layer

        well_index = well_id_list[image_i]
        time_index = time_id_list[image_i]
        fname_suffix = f"well{well_index:03}_t{time_index:03}"

        # load point cloud
        point_cloud_options = glob(os.path.join(fin_point_dir, "*" + fname_suffix + "*"))
        local_options = [p for p in point_cloud_options if "global" not in p]
        global_options = [p for p in point_cloud_options if "global" in p]
        if (len(local_options) > 1) | (len(global_options) > 1):
            raise Exception("Found multiple files with same well and time index.")
        point_df = pd.read_csv(local_options[0], index_col=0)
        global_df = pd.read_csv(global_options[0], index_col=0)
        point_df.loc[:, global_df.columns.tolist()] = np.tile(global_df.iloc[:, :].to_numpy(),
                                                                   (point_df.shape[0], 1)).copy()
        if "fin_label_cur" not in point_df.columns:
            point_df["fin_label_cur"] = -1
        # point_df["fin_label_pd"] = -1
        # load probability map from cellpose
        prob_file_list = glob(prob_file_dir + "*" + fname_suffix + "_probs*")
        if len(prob_file_list) > 1:
            raise Exception("Found multiple files with same well and time index.")
        im_prob = skio.imread(prob_file_list[0], plugin="tifffile")

        # generate temporary DF to keep track of curation labels
        curation_df = pd.DataFrame(point_df.loc[:, "fin_label_cur"].copy())
        curation_df["fin_label_pd"] = 0
        # 0=outlier, 1=fin, 2=yolk, 3=body
        curation_df["outlier_flags"] = curation_df["fin_label_cur"] == 0
        curation_df["fin_flags"] = curation_df["fin_label_cur"] == 1
        curation_df["yolk_flags"] = curation_df["fin_label_cur"] == 2
        curation_df["body_flags"] = curation_df["fin_label_cur"] == 3

        curation_df.loc[:, "fin_label_cur"] += 1  # for labeling convenience

        # generate wide feature DF for classifier training
        train_df_temp = point_df.loc[point_df["fin_label_cur"] != -1]
        train_df_temp.loc[:, "well_index"] = well_index
        train_df_temp.loc[:, "time_index"] = time_index
        # train_df_temp.loc[:, global_df.columns.tolist()] = np.tile(global_df.iloc[:, :].to_numpy(), (train_df_temp.shape[0], 1)).copy()
        if len(train_df) > 0:
            train_df = pd.concat([train_df, train_df_temp], axis=0)
            train_df = train_df.drop_duplicates(ignore_index=True)
        else:
            train_df = train_df_temp.copy()

        # initialize model if it doesn't already exist

        # set colormap
        label_color_cycle = ["white", "gray", "green", "red", "blue"]

        # initialize viewer
        viewer = napari.view_image(im_prob, colormap="gray", scale=scale_vec,
                                   contrast_limits=(point_df.loc[0, "prob_thresh"], np.percentile(im_prob, 99.8)))

        curation_df.loc[:, "fin_label_cur"] = curation_df.loc[:, "fin_label_cur"]/4
        curation_df.loc[:, "fin_label_pd"] = curation_df.loc[:, "fin_label_cur"].copy()
        # # generate master point array to integrate results
        point_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='point labels',
                                        size=4, features=curation_df.loc[:, "fin_label_cur"], face_color="fin_label_cur",
                                        face_colormap=lb_colormap, visible=True)


        outlier_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='outlier points',
                                        size=4, features=curation_df.loc[:, "outlier_flags"], face_color="outlier_flags",
                                        face_color_cycle=label_color_cycle[:2], visible=False)

        fin_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='fin points',
                                          size=4, features=curation_df.loc[:, "fin_flags"],
                                          face_color="fin_flags",
                                          face_color_cycle=[label_color_cycle[0]] + [label_color_cycle[2]], visible=False)

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
                                     face_colormap=lb_colormap, visible=True)

        # pd_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='predictions',
        #                                 size=4, features=curation_df.loc[:, "fin_label_pd"],
        #                                 face_color="fin_label_pd", opacity=0.7, visible=True)
        #
        # pd_layer.face_color_cycle_map = {
        #     0: "gray",
        #     1: "green",
        #     2: "red",
        #     3: "blue"
        # }
        # pd_layer.refresh_colors()

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
        # point_df.loc[:, "fin_label_cur"] = point_layer.features.copy() - 1

        napari.run()

        print("Saving...")

        # point_df.to_csv(local_options[0])


        wait = input(
            "Press Enter to continue to next image. \nPress 'x' then Enter to exit. \nPress Enter to move to next experiment.")
        if wait == 'x':
            break
        # elif wait == 'n':
        #     image_i += 1
        else:
            image_i += 1
        # print(image_i)
# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/"
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\"
    experiment_date = "20231214"
    overwrite = True
    curate_pec_fins(root, experiment_date, overwrite_flag=overwrite)

    napari.run()