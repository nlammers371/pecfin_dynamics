import numpy as np
import os
from glob2 import glob
from tqdm import tqdm
from src.build.build02_stitch_nuclear_masks import restitch_masks
from src.utilities.functions import path_leaf
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

experiment_date_vec = ["20240424", "20240425"]
seg_model_vec = ["tdTom-dim-log-v3", "tdTom-dim-log-v3"]

thresh_range = np.arange(-2, 6, 2)

for e, experiment_date in enumerate(experiment_date_vec):
    print("Processing " + experiment_date + "...")
    seg_model = seg_model_vec[e]
    # get list of zarr files
    mask_stack_list = sorted(glob(os.path.join(root, "built_data", "mask_stacks", seg_model, experiment_date, "*well0004_mask_stacks.zarr")))

    for m, zarr_path in enumerate(tqdm(mask_stack_list, "Restitching masks...")):
        fname = path_leaf(zarr_path)
        # get pob path
        prob_zarr_path = os.path.join(root, "built_data", "cellpose_output", seg_model, experiment_date, fname.replace("mask_stacks", "probs"))
        restitch_masks(zarr_path, prob_zarr_path, thresh_range)

