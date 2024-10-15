import napari
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
# from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
import scipy.ndimage as ndi
import skimage.io as io
from ultrack.config.config import load_config
from ultrack.core.export.tracks_layer import to_tracks_layer
from skimage.transform import resize
import json
import nd2

# # set parameters
# root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"


def load_tracking_data(root, experiment_date, well_num, config_name, model, register_images=True, start_i=None, stop_i=None, suffix=""):
    tracking_folder = config_name.replace(".txt", "")
    tracking_folder = tracking_folder.replace(".toml", "")

    # scale_vec = np.asarray([2.0, 0.55, 0.55])
    # centroid_flag = False
    # label_spacer = False

    # get path to metadata
    metadata_path = os.path.join(root, "metadata", "tracking")

    # path to image data
    data_path = os.path.join(root, "built_data", "cellpose_output", model, experiment_date, "")
    filename = experiment_date + f"_well{well_num:04}_probs.zarr"

    # load deconvolved image data
    image_path = os.path.join(data_path, filename)
    data_zarr = zarr.open(image_path, mode='r')
    scale_vec = data_zarr.attrs["voxel_size_um"]

    # apply registration to align raw data and tracks
    if "shift_array" in data_zarr.attrs.keys() and register_images:
        shift_array = np.asarray(data_zarr.attrs["shift_array"])
        data_plot = np.zeros_like(data_zarr)
        data_plot[0] = data_zarr[0]
        for t in tqdm(range(1, shift_array.shape[0]), "Applying registration to raw data..."):
            data_plot[t] = ndi.shift(data_zarr[t], (shift_array[t]), order=1)
    elif "shift_array" in data_zarr.attrs.keys():
        shift_array = np.asarray(data_zarr.attrs["shift_array"])
        data_plot = data_zarr
    else:
        shift_array = []
        data_plot = data_zarr

    # set path to tracking-related data
    if stop_i is None:
        stop_i = data_zarr.shape[0]
    if start_i is None:
        start_i = 0

    project_path = os.path.join(root, "tracking", experiment_date,  tracking_folder, f"well{well_num:04}", "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")

    # load tracking masks
    label_path = os.path.join(project_sub_path, "segments.zarr")
    seg_zarr = zarr.open(label_path, mode='r')

    # cfg = load_config(os.path.join(metadata_path, config_name))
    # tracks_df, graph = to_tracks_layer(cfg)
    tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))

    return data_plot, seg_zarr, tracks_df, shift_array, scale_vec

if __name__ == "__main__":
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date = "20240619"
    config_name = "tracking_jordao_20240918.txt"
    model = "tdTom-bright-log-v5"

    data_plot, seg_zarr, tracks_df, shift_array, scale_vec = load_tracking_data(root, experiment_date, well_num=2,
                                                          config_name=config_name, model=model, register_images=False)

    viewer = napari.view_image(data_plot, scale=tuple(scale_vec), contrast_limits=[-20, 40])

    viewer.add_tracks(
        tracks_df[["track_id", "t", "z", "y", "x"]],
        name="tracks",
        scale=tuple(scale_vec),
        translate=(0, 0, 0, 0),
        visible=False,
    )


    viewer.add_labels(
        seg_zarr,
        name="segments",
        scale=tuple(scale_vec),
        translate=(0, 0, 0, 0),
    ).contour = 2

    viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')

    napari.run()