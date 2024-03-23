from tqdm import tqdm
import numpy as np
import napari
import glob2 as glob
import os
import skimage.io as io
import nd2
from skimage.morphology import label
import zarr

# set paths
root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"

experiment_date = "20240223"
model1 = "bkg-v2"
label_directory1 = os.path.join(root, "built_data", "cellpose_output", model1, experiment_date, '')
model2 = "log-v5"
label_directory2 = os.path.join(root, "built_data", "cellpose_output", model2, experiment_date, '')
data_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')


# get list of images
image_list = sorted(glob.glob(data_directory + "*.zarr"))

# zarr_path = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
time_int = 201
well_int = 2

# load labels and probs
prob_name = experiment_date + f"_well{well_int:03}_t{time_int:03}_probs.tif"
prob_path1 = os.path.join(label_directory1, prob_name)
im_prob1 = io.imread(prob_path1)
label_name = experiment_date + f"_well{well_int:03}_t{time_int:03}_labels.tif"
label_path1 = os.path.join(label_directory1, label_name)
im_label1 = io.imread(label_path1)

prob_path2 = os.path.join(label_directory2, prob_name)
im_prob2 = io.imread(prob_path2)
label_path2 = os.path.join(label_directory2, label_name)
im_label2 = io.imread(label_path2)

# extract key metadata info
data_tzyx = zarr.open(image_list[well_int], mode="r")
data_zyx = data_tzyx[time_int]

dtype = data_zyx.dtype
scale_vec = tuple([2.0, 0.55, 0.55])


# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(data_zyx, scale=scale_vec)
viewer.add_labels(im_label1, scale=scale_vec, name="bkg")
viewer.add_labels(im_label2, scale=scale_vec, name="log")
# viewer.add_labels(label(im_prob > -4), scale=scale_vec)
# viewer.add_image(im_prob, scale=scale_vec)
# viewer.add_labels(label(im_prob>=2), scale=scale_vec, name="prob thresh labels") #, contrast_limits=[-16, 16])
# viewer.add_image(z_depth_stack, scale=scale_vec_plot, opacity=0.5, colormap="magma")
# viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(label_stack, scale=scale_vec_plot)
# movie = Movie(myviewer=viewer)
napari.run()


# if __name__ == '__main__':
#     napari.run()