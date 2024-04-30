import pandas as pd
import numpy as np
import napari
import glob2 as glob
import os
from src.utilities.functions import path_leaf
import zarr
from skimage.measure import regionprops
from scipy.spatial.distance import cdist

# root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/"
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\"
model_name = "log-v5"
experiment_date = "20240223"
well_ind = -1
time_int = 0
scale_vec = tuple([2.0, 0.55, 0.55])
# prob_thresh = -8

# get path to cellpose output
cellpose_directory = os.path.join(root, "built_data", "cellpose_output", model_name, experiment_date, '')
mask_directory = os.path.join(root, "built_data", "stitched_labels", model_name, experiment_date, '')

# open the probability zarr file
well_list = sorted(glob.glob(cellpose_directory + "*_probs.zarr"))
well = well_list[well_ind]
file_prefix = path_leaf(well).replace("_probs.zarr", "")

prob_name = os.path.join(cellpose_directory, file_prefix + "_probs.zarr")
mask_name = os.path.join(mask_directory, file_prefix + "_labels_stitched.zarr")

# extract key metadata info
prob_zarr = zarr.open(prob_name, mode="r")
mask_zarr = zarr.open(mask_name, mode="r")
im_prob = prob_zarr[time_int]
im_mask = mask_zarr[time_int]

# get image dims
im_dims = im_prob.shape

# get nucleus centroids
regions = regionprops(im_mask)
centroid_array = np.asarray([rg["Centroid"] for rg in regions])
centroid_array = np.multiply(centroid_array, np.asarray(scale_vec))
# load points dataset
# df_path = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/built_data/point_clouds/feature_predictions/20240223/20240223_well0012_time0000_centroids.csv"
df_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\built_data\\point_clouds\\feature_predictions\\20240223\\20240223_well0012_time0000_centroids.csv"
points_df = pd.read_csv(df_path, index_col=0)

fin_points = points_df.loc[points_df["fin_label_pd"] == 0.5, ["Z", "Y", "X"]].to_numpy()
# norm_factors = np.max(fin_points, axis=0)
# fin_point_norm = np.divide(fin_points, norm_factors)

# get distances to fin points
dist_mat = cdist(centroid_array, fin_points)
min_dists = np.min(dist_mat, axis=1)
lb_vec = np.asarray([rg["label"] for rg in regions])
fin_mask_indices = lb_vec[min_dists <= 3]
fin_mask_indices = fin_mask_indices[fin_mask_indices > 0]

# create new layer
ft_mat = np.isin(im_mask, fin_mask_indices)
fin_mask = im_mask.copy()
fin_mask[~ft_mat] = 0

fin_prob = im_prob.copy()
fin_prob[~ft_mat] = -16

# make cell-type examples
im_cond = im_prob.copy()
im_cond[im_mask != 3054] = np.nan

im_cond2 = im_prob.copy()
im_cond2[im_mask != 304] = np.nan

im_int = im_prob.copy()
im_int[im_mask != 534] = np.nan

im_int2 = im_prob.copy()
im_int2[im_mask != 4371] = np.nan

im_aer = im_prob.copy()
im_aer[im_mask != 4407] = np.nan

im_aer2 = im_prob.copy()
im_aer2[im_mask != 510] = np.nan

im_epi = im_prob.copy()
im_epi[im_mask != 679] = np.nan

im_epi2 = im_prob.copy()
im_epi2[im_mask != 975] = np.nan

# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(im_prob, scale=tuple(scale_vec), contrast_limits=[-10, 40])
viewer.add_image(fin_prob, scale=tuple(scale_vec), contrast_limits=[-10, 40])
viewer.add_labels(im_mask, scale=tuple(scale_vec))
viewer.add_labels(fin_mask, scale=tuple(scale_vec))

viewer.add_image(im_cond, scale=tuple(scale_vec), colormap="bop blue", contrast_limits=[-10, 20])
viewer.add_image(im_cond2, scale=tuple(scale_vec), colormap="bop blue", contrast_limits=[-10, 20])

viewer.add_image(im_int, scale=tuple(scale_vec), colormap="bop orange", contrast_limits=[-10, 20])
viewer.add_image(im_int2, scale=tuple(scale_vec), colormap="bop orange", contrast_limits=[-10, 20])

viewer.add_image(im_aer, scale=tuple(scale_vec), colormap="bop purple", contrast_limits=[-10, 20])
viewer.add_image(im_aer2, scale=tuple(scale_vec), colormap="bop purple", contrast_limits=[-10, 20])

viewer.add_image(im_epi, scale=tuple(scale_vec), colormap="I Bordeaux", contrast_limits=[-10, 20])
viewer.add_image(im_epi2, scale=tuple(scale_vec), colormap="I Bordeaux", contrast_limits=[-10, 20])
# fin_hull_layer = viewer.add_surface(surf, name="fin_hull", colormap="bop blue", opacity=0.5)
# viewer.add_image(z_depth_stack, scale=scale_vec_plot, opacity=0.5, colormap="magma")
viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(label_stack, scale=scale_vec_plot)
# movie = Movie(myviewer=viewer)
napari.run()


# if __name__ == '__main__':
#     napari.run()