from tqdm import tqdm
import numpy as np
import napari
import glob2 as glob
import os
import skimage.io as io
from naparimovie import Movie

# set paths
root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/built_data/cellpose_output/"
experiment_date = "20231214"
series_number = 15

# read in raw datafile
prob_list = sorted(glob.glob(os.path.join(root, experiment_date, f"*well{series_number-1:03}*_probs.tif")))
label_list = sorted(glob.glob(os.path.join(root, experiment_date, f"*well{series_number-1:03}*_labels.tif")))

scale_vec_plot = tuple([1]) + tuple([2.0, 0.275, 0.275])

print("Loading probability images...")
prob_images = []
for pf in tqdm(prob_list):
    im_temp = io.imread(pf)
    prob_images.append(im_temp[np.newaxis, :, :, :])

prob_stack = np.concatenate(tuple(prob_images), axis=0)

print("Loading labels...")
# label_images = []
# for lb in tqdm(label_list):
#     im_temp = io.imread(lb)
#     label_images.append(im_temp[np.newaxis, :, :, :])
#
# label_stack = np.concatenate(tuple(label_images), axis=0)

im_shape = im_temp.shape
# generate depth-encoded image
z_vec = np.arange(im_shape[0]) #* pixel_res_vec[0]
y_vec = np.arange(im_shape[1]) #* pixel_res_vec[1]
x_vec = np.arange(im_shape[2]) #* pixel_res_vec[2]

z_ref_array, y_ref_array, x_ref_array = np.meshgrid(z_vec, y_vec, x_vec, indexing="ij")

im_prob_thresh = prob_stack >= -4
z_depth_stack = np.multiply(im_prob_thresh, z_ref_array)
prob_stack[prob_stack > 32] = 32

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(prob_stack, scale=scale_vec_plot, contrast_limits=[-32, 34])
viewer.add_image(z_depth_stack, scale=scale_vec_plot, opacity=0.5, colormap="magma")
viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(label_stack, scale=scale_vec_plot)
# movie = Movie(myviewer=viewer)
napari.run()


# if __name__ == '__main__':
#     napari.run()