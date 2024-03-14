import napari
import os
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
nd2_path = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
experiment_date = "20240223"  #"231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
config_name = "tracking_v1.txt"
tracking_folder = config_name.replace(".txt", "")
tracking_folder = tracking_folder.replace(".toml", "")
well_num = 2

# get path to metadata
metadata_path = os.path.join(root, "metadata", "tracking")

# set output path for tracking results
project_path = os.path.join(root, "tracking", experiment_date,  tracking_folder, f"well{well_num:04}", "")

# path to image data
data_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, "")
filename = experiment_date + f"_well{well_num:04}.zarr"
image_zarr = os.path.join(data_path, filename)

label_zarr = os.path.join(project_path, "segments.zarr")

# metadata_file_path = os.path.join(root, "metadata", experiment_date, "metadata.json")
# f = open(metadata_file_path)
# metadata = json.load(f)
# scale_vec = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])
# scale_vec_im = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])
imObject = nd2.ND2File(nd2_path)
res_raw = imObject.voxel_size()
scale_vec = np.asarray(res_raw)[::-1]


data_tzyx = zarr.open(image_zarr, mode='r')
segments = zarr.open(label_zarr, mode='r')



# viewer.add_labels(label_tzyx, scale=tuple(scale_vec), name="raw labels")

cfg = load_config(os.path.join(project_path, config_name))
tracks_df, graph = to_tracks_layer(cfg)

viewer = napari.view_image(data_tzyx[:25, :, 300:700, :], scale=tuple(scale_vec))

# tracks_df_ft = tracks_df.loc[(tracks_df["t"] >= start_i) & (tracks_df["t"] < stop_i), :]
#
#
# track_index = np.unique(tracks_df_ft["track_id"])
# keys = graph.keys()
# graph_ft = {k:graph[k] for k in keys if k in track_index}
viewer.add_tracks(
    tracks_df[["track_id", "t", "z", "y", "x"]],
    name="tracks",
    graph=graph,
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
    visible=False,
)

# segments = zarr.open(os.path.join(save_directory, "segments.zarr"), mode='r')

viewer.add_labels(
    segments,
    name="segments",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
).contour = 2


if __name__ == '__main__':
    napari.run()