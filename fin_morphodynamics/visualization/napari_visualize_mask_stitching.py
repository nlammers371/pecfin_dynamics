from tqdm import tqdm
import numpy as np
import napari
import glob2 as glob
import os
import skimage.io as io
import nd2
from skimage.morphology import label
from skimage.measure import regionprops
import zarr
from scipy.spatial import KDTree
import skimage.graph as graph
from skimage.transform import resize
import skimage as ski
import SimpleITK as sitk
import networkx as nx
import pandas as pd
from os.path import isdir, join
from os import makedirs


# set paths
root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"

experiment_date = "20240223"
model2 = "log-v5"
label_directory = os.path.join(root, "built_data", "cellpose_output", model2, experiment_date + "_stitched", '')


# zarr_path = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
time_int = 201
well_int = 2

# load labels and probs
out_prefix = experiment_date + f"_well{well_int:03}_t{time_int:03}"

cp_mask = io.imread(join(label_directory, out_prefix + "_cp_mask.tif"))
stitched_mask = io.imread(join(label_directory, out_prefix + "_aff_mask.tif"))
stitched_mask_0_flow10 = io.imread(join(label_directory, out_prefix + "_aff_mask0_flow10.tif"))
im_bkg = io.imread(join(label_directory, out_prefix + "_im_bkg.tif"))
im_LoG = io.imread(join(label_directory, out_prefix + "_im_LoF.tif"))

scale_vec = tuple([2.0, 0.55, 0.55])


# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(im_bkg)
# viewer.add_labels(im_label1, scale=scale_vec, name="bkg")
viewer.add_labels(cp_mask, name="cp labels")
viewer.add_labels(stitched_mask, name="aff labels")
viewer.add_labels(stitched_mask_0_flow10, name="aff labels flow 10")

# viewer.add_labels(label(im_prob > -4), scale=scale_vec)


# if __name__ == '__main__':
#     napari.run()