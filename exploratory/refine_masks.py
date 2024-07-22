import os
import numpy as np
import glob2 as glob
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops
import napari
import zarr
from src.utilities.functions import path_leaf
from sklearn.neighbors import KDTree
import scipy

well_index = 3
time_index = 140

root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
experiment_date = "20240619"
model_name = "tdTom-bright-log-v5"

# get directory to stitched labels
mask_directory = os.path.join(root, "built_data", "mask_stacks", model_name, experiment_date, '')
prob_directory = os.path.join(root, "built_data", "cellpose_output", model_name, experiment_date, '')

# get list of wells with labels to stitch
well_list = sorted(glob.glob(mask_directory + "*_mask_aff.zarr"))

well = well_list[well_index]
mask_zarr = zarr.open(well, mode='r')

prob_zarr_name = path_leaf(well).replace("_mask_aff", "_probs")
stack_zarr_name = path_leaf(well).replace("_mask_aff", "_mask_stacks")
prob_zarr = zarr.open(os.path.join(prob_directory, prob_zarr_name), mode='r')
stack_zarr = zarr.open(os.path.join(mask_directory, stack_zarr_name), mode='r')

im_prob = prob_zarr[time_index, :, :-300, :]
im_mask = mask_zarr[time_index, :, :-300, :]
im_stack = stack_zarr[time_index, :, :, :-300, :]

# try simple global threshold
thresh = 0
im_mask_thresh = im_mask.copy()
im_mask_thresh[im_prob < thresh] = 0

# nucleus-specific thresholding
regions = regionprops(im_mask, intensity_image=np.exp(im_prob / 15))
centroids = np.asarray([rg["centroid"] for rg in regions])
centroids_wt = np.asarray([rg["centroid_weighted"] for rg in regions])
im_mask_local_thresh = np.zeros_like(im_mask)
for rg in tqdm(regions):
    indices = tuple(rg.coords.T) #np.where(im_mask == rg.label)
    p_vec = im_prob[indices]
    thresh = np.percentile(p_vec, 75)
    thresh_ft = p_vec >= thresh

    indices_thresh = list(indices)
    indices_thresh[0] = indices_thresh[0][thresh_ft]
    indices_thresh[1] = indices_thresh[1][thresh_ft]
    indices_thresh[2] = indices_thresh[2][thresh_ft]
    indices_thresh = tuple(indices_thresh)

    im_mask_local_thresh[indices_thresh] = rg.label


# extract useful info
scale_vec = mask_zarr.attrs["voxel_size_um"]

viewer = napari.view_image(im_prob, scale=scale_vec)
# viewer.add_image(np.exp(im_prob / 15), scale=scale_vec)
viewer.add_labels(im_mask, scale=scale_vec, opacity=0.8)
viewer.add_points(centroids_wt, name='centroids_wt', scale=scale_vec)
viewer.add_points(centroids, name='centroids', scale=scale_vec)
# viewer.add_labels(im_stack[0], scale=scale_vec, opacity=0.8)
# viewer.add_labels(im_mask_local_thresh, scale=scale_vec, opacity=0.4, name="local")
# viewer.add_labels(im_mask_thresh, scale=scale_vec, opacity=0.4, name="thresholded")

