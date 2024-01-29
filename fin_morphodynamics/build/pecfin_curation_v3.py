# from aicsimageio import AICSImage
import numpy as np
import napari
import os
from glob2 import glob
import skimage.io as skio
from alphashape import alphashape
from functions.utilities import path_leaf
from skimage.transform import resize
import pandas as pd
# import open3d as o3d


def on_points_click(layer, event):
    if event.type == 'mouse_press' and event.button == 1 and layer.mode == "select":  # Left mouse button
        # selected_index = layer.get_value(event.position, world=True)
        # print(f"Selected point index: {selected_index}")
        selected_index = layer.get_value(event.position, world=True, view_direction=event.view_direction,
                                         dims_displayed=event.dims_displayed)
        if selected_index is not None:
            layer_features = layer.features.copy()
            layer_features.iloc[selected_index, 0] = ~layer.features.iloc[selected_index, 0]
            layer.features = layer_features
        else:
            tp = np.array(event.position)
            ld = layer._view_data
            distances = np.sqrt(np.sum((tp-ld)**2, axis=1))
            selected_index = np.argmin(distances)

            layer_features = layer.features.copy()
            layer_features.iloc[selected_index, 0] = ~layer.features.iloc[selected_index, 0]
            layer.features = layer_features

        # if layer.name == 'outlier points':
        #     selected_index = layer.get_value(event.position, world=True)
        #     print("outlier")
        # elif layer.name == 'fin points':
        #     selected_index = layer.get_value(event.position, world=True)
        #     print("fin")
        # elif layer.name == 'yolk points':
        #     point_data = event.source.data
        #     print("yolk")
        # elif layer.name == 'body points':
        #     point_data = event.source.data
        #     print("body")
        return

def event_trigger(event):
    if event.source.name == 'outlier points':
        point_data = event.source.data
        print("outlier")
    elif event.source.name == 'fin points':
        point_data = event.source.data
        print("fin")
    elif event.source.name == 'yolk points':
        point_data = event.source.data
        print("yolk")
    elif event.source.name == 'body points':
        point_data = event.source.data
        print("body")

# 
# def calculate_fin_hull(point_data, hull_alpha=4):
# 
#     # point_data = point_data.source.data
#     point_data_norm = np.divide(point_data, im_dims)
#     update_lb_flag = False
#     if point_data.shape[0] == 6:
#         update_lb_flag = True
#         
#         hull = alphashape(point_data_norm, alpha=hull_alpha)
#         values = np.linspace(0.5, 1, len(hull.vertices))
#         surf = (np.multiply(hull.vertices, im_dims), hull.faces, values)
#         fin_hull_layer = viewer.add_surface(surf, name="fin_hull", colormap="bop blue", scale=scale_vec, opacity=0.5)
# 
#     elif point_data.shape[0] > 6:
#         update_lb_flag = True
#         
#         hull = alphashape(point_data_norm, alpha=hull_alpha)
#         values = np.linspace(0.5, 1, len(hull.vertices))
#         surf = (np.multiply(hull.vertices, im_dims), hull.faces, values)
# 
#         if "fin_hull" not in viewer.layers:
#             fin_hull_layer = viewer.add_surface(surf, name="fin_hull", colormap="bop blue", scale=scale_vec,
#                                                 opacity=0.5)
#         viewer.layers["fin_hull"].data = surf
#         viewer.layers["fin_hull"].refresh()
#     
#     if update_lb_flag:
#         if len(hull.faces) > 2:
#             inside_flags = hull.contains(zyx_nuclei_norm)
#             fin_points = zyx_nuclei[np.where(inside_flags == 1)[0], :].astype(int)
#             new_labels = np.zeros(im_dims_ds, dtype=np.int8)
#             new_labels[fin_points[:, 0], fin_points[:, 1], fin_points[:, 2]] = 1
#             viewer.layers["labels"].data = resize(new_labels, im_dims, preserve_range=True, order=0)
#             viewer.layers["labels"].refresh()

def curate_pec_fins(root, experiment_date, overwrite_flag=False):
    
    # load metadata
    metadata_path = os.path.join(root, "metadata", experiment_date + "_master_metadata_df.csv")
    metadata_df = pd.read_csv(metadata_path, index_col=0)
    
    # add column to track fin segmentation
    if "fin_curation_flag" not in metadata_df.columns:
        metadata_df["fin_curation_flag"] = False
    if "fin_curation_date" not in metadata_df:
        metadata_df["fin_curation_date"] = np.nan
    
    # load raw image
    # ref_image_path = glob(os.path.join(root, "raw_data", experiment_date, "*.nd2"))
    # if len(ref_image_path) > 1:
    #     raise Exception("Multiple nd2 files found.")
    #
    # imObject = AICSImage(ref_image_path[0])
    
    # git list of prob files produced by CellPose
    prob_file_dir = os.path.join(root, "built_data", "cellpose_output", experiment_date, "")
    
    # make directory for saving fin masks
    fin_point_dir = os.path.join(root, "built_data", "point_clouds", experiment_date, "")
    
    # Extract pixel sizes and bit_depth
    pixel_res_vec = metadata_df.loc[0, ["z_res_um", "y_res_um", "x_res_um"]].to_numpy().astype(np.float32)
    scale_vec = tuple(pixel_res_vec)
    scale_array = np.asarray(scale_vec)
    
    # get list of fins to segment
    if not overwrite_flag:
        seg_flags = metadata_df["fin_curation_flag"] == False
        well_id_list = metadata_df.loc[seg_flags, "well_index"]
        time_id_list = metadata_df.loc[seg_flags, "time_index"]
    else:
        well_id_list = metadata_df.loc[:, "well_index"]
        time_id_list = metadata_df.loc[:, "time_index"]
    
    image_i = 0

    while image_i < len(well_id_list):
        
        fname_suffix = f"well{well_id_list[image_i]:03}_t{time_id_list[image_i]:03}"
        
        # load point cloud
        point_cloud_options = glob(os.path.join(fin_point_dir, "*" + fname_suffix + "*"))
        if len(point_cloud_options) > 1:
            raise Exception("Found multiple files with same well and time index.")
        point_df = pd.read_csv(point_cloud_options[0], index_col=0)
        if "fin_label" not in point_df.columns:
            point_df["fin_label_cur"] = -1

        # load probability map from cellpose
        prob_file_list = glob(prob_file_dir + "*" + fname_suffix + "_probs*")
        if len(prob_file_list) > 1:
            raise Exception("Found multiple files with same well and time index.")
        im_prob = skio.imread(prob_file_list[0], plugin="tifffile")

        # generate temporary DF to keep track of curation labels
        curation_df = pd.DataFrame(point_df.loc[:, "fin_label_cur"].copy())
        # 0=outlier, 1=fin, 2=yolk, 3=body
        curation_df["outlier_flags"] = curation_df["fin_label_cur"] == 0
        curation_df["fin_flags"] = curation_df["fin_label_cur"] == 1
        curation_df["yolk_flags"] = curation_df["fin_label_cur"] == 2
        curation_df["body_flags"] = curation_df["fin_label_cur"] == 3

        # set colormap
        label_color_cycle = ["white", "gray", "green", "red", "blue"]

        # initialize viewer
        viewer = napari.Viewer()#napari.view_image(im_prob, colormap="gray", scale=scale_vec,
                                   #contrast_limits=(point_df.loc[0, "prob_thresh"], np.percentile(im_prob, 99.8)))

        # # generate master point array to integrate results
        # point_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='point labels',
        #                                 size=4, features=curation_df.loc[:, "fin_label_cur"], face_color="fin_label_cur",
        #                                 face_color_cycle=label_color_cycle, visible=True)

        outlier_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='outlier points',
                                        size=4, features=curation_df.loc[:, "outlier_flags"], face_color="outlier_flags",
                                        face_color_cycle=label_color_cycle[:2], visible=False)

        # fin_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='fin points',
        #                                   size=4, features=curation_df.loc[:, "fin_flags"],
        #                                   face_color="fin_flags",
        #                                   face_color_cycle=[label_color_cycle[0]] + [label_color_cycle[2]], visible=False)
        #
        # yolk_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='yolk points',
        #                               size=4, features=curation_df.loc[:, "yolk_flags"],
        #                               face_color="yolk_flags",
        #                               face_color_cycle=[label_color_cycle[0]] + [label_color_cycle[3]], visible=False)
        #
        # body_layer = viewer.add_points(point_df.loc[:, ["Z", "Y", "X"]].to_numpy(), name='body points',
        #                                size=4, features=curation_df.loc[:, "body_flags"],
        #                                face_color="body_flags",
        #                                face_color_cycle=[label_color_cycle[0]] + [label_color_cycle[4]], visible=False)

        # connect to event trigger function
        outlier_layer.mouse_drag_callbacks.append(on_points_click)
        # fin_layer.mouse_click_callbacks.append(on_points_click)
        # yolk_layer.mouse_click_callbacks.append(on_points_click)
        # body_layer.mouse_click_callbacks.append(on_points_click)

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

        napari.run()

        # points_layer = viewer.layers["fin hull points"]
        # label_array = np.asarray(viewer.layers["labels"].data)



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