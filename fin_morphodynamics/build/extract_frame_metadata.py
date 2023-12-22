"""
Image segmentation via Cellpose library
"""
from tifffile import TiffWriter
from tqdm import tqdm
import logging
import glob2 as glob
import os
import time
from aicsimageio import AICSImage
from typing import Any
from typing import Dict
from functions.utilities import path_leaf
import numpy as np

def extract_metadata(
    *,
    # Fractal arguments
    root: str,
    experiment_date: str,
    n_tres_frames: int = 10
) -> Dict[str, Any]:


    raw_directory = os.path.join(root, "raw_data", experiment_date, '')
    # if tiff

    save_directory = os.path.join(root, "metadata", "frame_metadata", '')
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # get list of images
    image_list = sorted(glob.glob(raw_directory + "*.nd2"))

    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    for im in range(len(image_list)):
        nd2_path = image_list[im]
        im_name = path_leaf(nd2_path)

        print("processing " + im_name)

        # read the image data
        imObject = AICSImage(nd2_path)

        # use first 10 frames to infer time resolution

        n_wells = len(imObject.scenes)
        well_list = imObject.scenes
        n_time_points = imObject.dims["T"][0]

        # extract key image attributes
        channel_names = imObject.channel_names  # list of channels and relevant info


    return {}

if __name__ == "__main__":
    # sert some hyperparameters
    overwrite = False
    model_type = "nuclei"
    output_label_name = "td-Tomato"
    seg_channel_label = "561"
    xy_ds_factor = 2

    # set path to CellPose model to use
    pretrained_model = "C:\\Users\\nlammers\\Projects\\pecfin_dynamics\\fin_morphodynamics\\cellpose_models\\nuclei_3D_gen_v1"

    # set read/write paths
    root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
    experiment_date = "20231013"

    cellpose_segmentation(root=root, experiment_date=experiment_date,
                          # raw_directory=raw_directory, save_data_directory=save_directory,
                          seg_channel_label=seg_channel_label, return_probs=True, xy_ds_factor=xy_ds_factor,
                          output_label_name=output_label_name, pretrained_model=pretrained_model, overwrite=overwrite)
