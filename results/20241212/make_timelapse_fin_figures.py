# from aicsimageio import AICSImage
import numpy as np
import napari
import os
from src.utilities.functions import path_leaf
from glob2 import glob
import pandas as pd
import zarr
from src.utilities.fin_class_def import FinData
from scipy.spatial import distance_matrix
from src.utilities.surface_axes_functions import *
from skimage import filters
from tqdm import tqdm

def get_curation_metadata(root, experiment_date, force_update=False):
    metadata_path = os.path.join(root, "metadata", "frame_metadata", experiment_date + "_master_metadata_df.csv")
    metadata_df = pd.read_csv(metadata_path)
    curation_dir = os.path.join(root, "metadata", "curation_metadata", "")
    if not os.path.exists(curation_dir):
        os.makedirs(curation_dir)
    if force_update or (not os.path.isfile(curation_dir + experiment_date + "_curation_metadata_df.csv")):  # this means we need to generate curation info
        curation_df = metadata_df.copy()
        curation_cols = ["has_point_features", "has_manual_tissue_labels", "fin_axis_approved", "body_axis_approved",
                         "tissue_labels_approved"]
        curation_df.loc[:, curation_cols] = False

        # check for point feature data
        point_feat_path = os.path.join(root, "point_cloud_data", "point_features", seg_type_global, "")
        for i in range(curation_df.shape[0]):
            well_num = curation_df.loc[i, "well_index"]
            time_int = curation_df.loc[i, "time_index"]
            point_prefix = experiment_date + f"_well{well_num:04}" + f"_time{time_int:04}"

            fl = glob(point_feat_path + point_prefix + "*")
            curation_df.loc[i, "has_point_features"] = len(fl) > 0

        # check manual tissue curation
        manual_curation_path = os.path.join(root, "point_cloud_data", "manual_curation", "")
        dir_list = sorted(glob(manual_curation_path + "*"))
        labels_list = []
        for dir_path in dir_list:
            labels_list += glob(os.path.join(dir_path, "*_labels.csv"))
        df_names = [path_leaf(lb)[:-11] for lb in labels_list]
        for i in range(curation_df.shape[0]):
            well_num = curation_df.loc[i, "well_index"]
            time_int = curation_df.loc[i, "time_index"]
            point_prefix = experiment_date + f"_well{well_num:04}" + f"_time{time_int:04}"

            curation_df.loc[i, "has_manual_tissue_labels"] = point_prefix in df_names

        # now, check approval status
        fin_object_path = os.path.join(root, "point_cloud_data", "fin_objects", "")
        for i in range(curation_df.shape[0]):
            well_num = curation_df.loc[i, "well_index"]
            time_int = curation_df.loc[i, "time_index"]
            point_prefix = experiment_date + f"_well{well_num:04}" + f"_time{time_int:04}"
            object_path = os.path.join(fin_object_path, point_prefix + "_fin_object.pkl")
            if not os.path.exists(object_path):
                curation_df.loc[i, "fin_axis_approved"] = False
                curation_df.loc[i, "body_axis_approved"] = False
                curation_df.loc[i, "tissue_labels_approved"] = False
            else:
                fin_data = FinData(data_root=root, name=point_prefix, tissue_seg_model=seg_type_global)
                curation_df.loc[i, "fin_axis_approved"] = fin_data.axis_fin_approved
                curation_df.loc[i, "body_axis_approved"] = fin_data.axis_body_approved
                curation_df.loc[i, "tissue_labels_approved"] = fin_data.seg_approved

        # save curation df
        curation_df.to_csv(curation_dir + experiment_date + "_curation_metadata_df.csv", index=False)
    else:
        curation_df = pd.read_csv(curation_dir + experiment_date + "_curation_metadata_df.csv")

    return curation_df, curation_dir + experiment_date + "_curation_metadata_df.csv"

def get_fin_layers(labels_df, prob_zarr, mask_zarr, pd_mask, dist_thresh=60):

    xyz_array_fin = labels_df.loc[labels_df["fin_label_curr"] == 1, ["X", "Y", "Z"]].to_numpy()
    xyz_array = labels_df.loc[:, ["X", "Y", "Z"]].to_numpy()

    if np.any(xyz_array_fin):
        dist_mat = distance_matrix(xyz_array, xyz_array_fin)
        fin_dist = np.min(dist_mat, axis=1)
    else:
        fin_dist = np.zeros((xyz_array.shape[0], 1)) #, xyz_array_fin)

      # just using simple euclidean distance for now

    # filter for nearby surface nuclei
    dist_ft = fin_dist <= dist_thresh
    labels_df.loc[:, "fin_proximity_flag"] = dist_ft
    close_ids = labels_df.loc[dist_ft, "nucleus_id"].to_numpy()

    # get subset mask
    label_mask_fin = np.zeros_like(mask_zarr)
    id_mask = np.isin(mask_zarr, close_ids)
    label_mask_fin[id_mask] = pd_mask[id_mask]

    # get filtered zarr
    prob_zarr_fin = np.zeros_like(prob_zarr)
    prob_zarr_fin = prob_zarr_fin + np.min(prob_zarr)
    for t in range(prob_zarr_fin.shape[0]):
        prob_zarr_fin[t][id_mask] = prob_zarr[t][id_mask]

    # make region-specific mask
    mask_zarr_fin = np.zeros_like(mask_zarr)
    mask_zarr_fin[id_mask] = mask_zarr[id_mask]

    return label_mask_fin, prob_zarr_fin, mask_zarr_fin

def load_zarr_data(root, seg_model, experiment_date, file_prefix, start_time, stop_time):

    # path to raw data
    # raw_zarr_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, file_prefix + ".zarr")
    mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, experiment_date, file_prefix + "_mask_aff.zarr")
    prob_zarr_path = os.path.join(root, "built_data", "cellpose_output", seg_model, experiment_date,
                                  file_prefix + "_probs.zarr")

    # data_zarr = zarr.open(raw_zarr_path, mode="r")
    mask_zarr = zarr.open(mask_zarr_path, mode="r")
    prob_zarr = zarr.open(prob_zarr_path, mode="r")

    # convert scale vec to tuple
    scale_vec = prob_zarr.attrs["voxel_size_um"]
    scale_vec = tuple(scale_vec)

    # load the specific time point we want
    # time_range = np.arange(np.max([0, time_int-3]), np.min([len(prob_zarr), time_int+4]))
    im_prob = prob_zarr[start_time:stop_time]
    im_mask = mask_zarr[start_time:stop_time]

    return im_prob, im_mask, scale_vec


def load_fin_object(root, file_prefix, time_int, seg_type):

    point_prefix = file_prefix + f"_time{time_int:04}"
    # check to see if fin class object exists
    fin_data = FinData(data_root=root, name=point_prefix, tissue_seg_model=seg_type)

    return fin_data


def animate_pec_fins(root, well_num, start_time, stop_time, experiment_date, seg_model, seg_type,
                     anchor_time=None, label_opacity=0.5):


    if anchor_time is None:
        anchor_time = stop_time - 1#int(np.mean([start_time, stop_time]))
        anchor_i = anchor_time - start_time

    # initialize global variables
    global seg_type_global
    # global mask_zarr_fin, label_mask, mask_zarr_fin, label_layer_fin, viewer, fin_point_layer, \
    #     seg_type_global, fin_data, fin_surface_layer, \
    #     prob_layer_all, prob_layer_fin, continue_curation, global_df_ind, label_layer

    seg_type_global = seg_type

    # check curation metadata. If there is no curation metadata file, create one
    # curation_df, curation_path = get_curation_metadata(root, experiment_date, True)


    point_prefix = experiment_date + f"_well{well_num:04}" + f"from_{start_time:04} to {stop_time:04}"
    print(f"Loading curation data for {point_prefix}...")
    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"

    prob_zarr, mask_zarr, scale_vec = load_zarr_data(root, seg_model, experiment_date, file_prefix, start_time, stop_time)

    # load point features and labels
    print("Loading labels...")
    processed_data_path = os.path.join(root, "point_cloud_data", "processed_fin_data", "")
    # df
    df_name = file_prefix + f"_time{anchor_time:04}_fin_data.csv"
    fin_df = pd.read_csv(processed_data_path + df_name)
    fin_nucleus_ids = fin_df["nucleus_id"].to_numpy()

    # show tissue labels for reference
    # labels_df = fin_data.full_point_data
    print("Generating fin mask...")
    pd_mask = np.zeros(mask_zarr[anchor_i].shape, dtype=np.float32)
    pd_mask[np.isin(mask_zarr[anchor_i], fin_nucleus_ids)] = 1

    # get fin-specific layers
    print("Generating float fin mask...")
    prob_zarr_fin0 = prob_zarr.copy()
    pd_mask_float = np.multiply(prob_zarr[anchor_i], pd_mask)
    pd_mask_float_sm = filters.gaussian(pd_mask_float, sigma=(7, 23, 23), mode='nearest')

    # gauss-blurred version
    prob_zarr_fin = np.multiply(prob_zarr_fin0, pd_mask_float_sm[None, :, :, :])
    i_vec = np.linspace(0.25, 1, prob_zarr_fin.shape[0])
    prob_zarr_fin = np.multiply(prob_zarr_fin, i_vec[:, None, None, None])

    # initialize viewer
    viewer = napari.Viewer(ndisplay=3)
    prob_layer_all = viewer.add_image(prob_zarr, colormap="gray", name="probabilities",
                     scale=scale_vec, contrast_limits=(-4, np.percentile(prob_zarr, 99.8)))

    prob_layer_fin = viewer.add_image(prob_zarr_fin, colormap="green", name="fin probs (1)",
                                      scale=scale_vec, contrast_limits=(-4, 100))

    prob_layer_fin2 = viewer.add_image(pd_mask_float, colormap="gray", name="fin probs (2)",
                                      scale=scale_vec, contrast_limits=(-4, 100))

    # Enable the scale bar
    viewer.scale_bar.visible = True

    # Customize the scale bar
    viewer.scale_bar.unit = "µm"  # Set unit (optional)
    viewer.scale_bar.position = "bottom_right"  # Options: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    viewer.scale_bar.colored = False  # False for white, True to match layer color
    viewer.scale_bar.font_size = 12

    # # Create a text overlay for the time step
    # text_overlay = viewer.add_text("", color="white", anchor="upper_left")
    #
    # # Update the text overlay based on the current time step
    # def update_time_step(event):
    #     time_step = int(viewer.dims.current_step[0])  # Get the current time step
    #     text_overlay.text = f"{np.round(22 + (10*time_step) / 60, 1)}"
    #
    # # Attach the update function to the dims events
    # viewer.dims.events.current_step.connect(update_time_step)

    # make fin-focused probability later
    # prob_layer_fin = viewer.add_image(pd_mask_float, name="probabilities (fin region)", colormap="gray", scale=scale_vec, visible=False)
    # prob_layer_fin = viewer.add_image(pd_mask_float_sm, name="probabilities (fin region)", colormap="gray",
    #                                   scale=scale_vec, visible=False)
    # show tissue predictions
    # viewer.add_labels(mask_zarr, scale=scale_vec, name='nuclei masks (static)', opacity=1,
    #                       visible=False)

    # label_layer = viewer.add_image(pd_mask[None, :, :, :], scale=scale_vec, colormap="Blues",
    #                 name='tissue predictions (static)', opacity=label_opacity, visible=True)
    # label_layer_fin = viewer.add_labels(label_mask_fin, scale=scale_vec, name='tissue predictions (fin region)', opacity=label_opacity,
    #                                 visible=False)


    # fin_point_layer = viewer.add_points(fin_data.full_point_data.loc[ft, ["Z", "Y", "X"]].to_numpy(),
    #                                     name='fin points', size=6, features=fin_data.full_point_data.loc[ft, ["Z", "Y", "X"]],
    #                                     face_color='Z', face_colormap="viridis", visible=False, out_of_slice_display=True)

    # Add the animation plugin to the viewer
    viewer.window.add_plugin_dock_widget(plugin_name="napari-animation")


    napari.run()


    # local_ind += 1



# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    # root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date = "20240619"  # "20240712_01"
    overwrite = True
    fluo_flag = False
    use_model_priors = True
    show_approved_frames = False
    seg_model = "tdTom-bright-log-v5" #"tdTom-bright-log-v5"  # "tdTom-dim-log-v3"
    # point_model = "point_models_pos"

    well_num = 2  # None
    start_time = 99  # None
    stop_time = 101
    animate_pec_fins(root, experiment_date=experiment_date, well_num=well_num, seg_type="tissue_only_best_model_tissue", #
                    seg_model=seg_model, start_time=start_time, stop_time=stop_time)


