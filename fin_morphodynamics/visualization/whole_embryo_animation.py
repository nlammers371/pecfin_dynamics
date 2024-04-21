import pandas as pd
from aicsimageio import AICSImage
import numpy as np
import napari
import glob2 as glob
import os
import alphashape
from fin_morphodynamics.src.utilities.functions import path_leaf
import zarr
from skimage.measure import regionprops
from scipy.spatial.distance import cdist

# filepath = "/Volumes/Sequoia/20230622/24hpf_tdTom_whole_embryo_better_2023_06_22__18_43_26_564.czi"
# imObject = AICSImage(filepath)
# imData = imObject.data
# # get pixel scales
# res_raw = imObject.physical_pixel_sizes
# scale_vec = tuple(np.asarray(res_raw))
#
# # generate napari viewer instance
# i_max = np.max(np.squeeze(imData), axis=0)
# viewer = napari.Viewer(ndisplay=2)
# viewer.add_image(i_max, colormap="magenta")
# # viewer.add_image(imData, scale=tuple(scale_vec))
# viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# # viewer.add_labels(label_stack, scale=scale_vec_plot)
# # movie = Movie(myviewer=viewer)
# napari.run()
root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
model_name = "log-v5"
experiment_date = "20240223"

cellpose_directory = os.path.join(root, "built_data", "cellpose_output", model_name, experiment_date, '')



# get list of wells with labels to stitch
well_list = sorted(glob.glob(cellpose_directory + "*_probs.zarr"))
well = well_list[0]
#########
file_prefix = path_leaf(well).replace("_probs.zarr", "")
print("Stitching data from " + file_prefix)
prob_name = os.path.join(cellpose_directory, file_prefix + "_probs.zarr")
grad_name = os.path.join(cellpose_directory, file_prefix + "_grads.zarr")
# mask_name = os.path.join(cellpose_directory, file_prefix + "_labels.zarr")

# mask_zarr = zarr.open(mask_name, mode="r")
prob_zarr = zarr.open(prob_name, mode="r")

viewer = napari.view_image(prob_zarr[-1], scale=tuple([2.0, 0.55, 0.55]))
viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# get number of time points
n_time_points = prob_zarr.shape[0]



# if __name__ == '__main__':
#     napari.run()