import napari
import os

import pandas as pd
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
import skimage.io as io
from skimage.transform import resize
import json
import nd2

# # set parameters
root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
experiment_date = "20240223"
config_name = "tracking_thresh.txt"
tracking_folder = config_name.replace(".txt", "")
tracking_folder = tracking_folder.replace(".toml", "")

well_num = 2
start_i = 0
stop_i = 50
scale_vec = np.asarray([2.0, 0.55, 0.55])
centroid_flag = False
label_spacer = False

suffix = ""
if centroid_flag:
    suffix = "_centroid"

if label_spacer:
    suffix += "_spacer"

# get path to metadata
metadata_path = os.path.join(root, "metadata", "tracking")

# set output path for tracking results
project_path = os.path.join(root, "tracking", experiment_date,  tracking_folder, f"well{well_num:04}" + suffix, "")

# path to image data
data_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, "")
filename = experiment_date + f"_well{well_num:04}.zarr"

# load tracking results
image_path = os.path.join(data_path, filename)
label_path = os.path.join(project_path, "segments.zarr")
data_zarr = zarr.open(image_path, mode='r')
data_zarr = data_zarr[start_i:stop_i, :, 400:850, 80:520]
seg_zarr = zarr.open(label_path, mode='r')

cfg = load_config(os.path.join(metadata_path, config_name))
# tracks_df, graph = to_tracks_layer(cfg)
tracks_df = pd.read_csv(os.path.join(project_path, "tracks.csv"))

viewer = napari.view_image(data_zarr, scale=tuple(scale_vec))


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


if __name__ == '__main__':
    napari.run()