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
import pandas as pd

import numpy as np

def extract_frame_metadata(
    *,
    # Fractal arguments
    root: str,
    experiment_date: str,
    n_tres_frames: int = 10,
    sheet_names = None
) -> Dict[str, Any]:

    if sheet_names is None:
        sheet_names = ["series_number_map", "genotype", "age"]

    row_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    col_nums = [i for i in range(12)]

    raw_directory = os.path.join(root, "raw_data", experiment_date, '')
    plate_directory = os.path.join(root, "metadata", "plate_maps", experiment_date + "_plate_map.xlsx")

    save_directory = os.path.join(root, "metadata", "frame_metadata", '')
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # get list of images
    image_list = sorted(glob.glob(raw_directory + "*.nd2"))

    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    if len(image_list) > 1:
        raise Exception("Multiple .nd2 files were found in target directory. Make sure to put fullembryo images into a subdirectory")

    nd2_path = image_list[0]
    im_name = path_leaf(nd2_path)
    well_coord_list = []
    for row in row_letters:
        for col in col_nums:
            well_coord_list.append(row + f"{col:02}")
    print("processing " + im_name)

    xl_temp = pd.ExcelFile(plate_directory)

    for sheet in sheet_names:
        sheet_temp = xl_temp.parse(sheet)  # read a specific sheet to DataFrame
        sheet_ravel = sheet_temp.iloc[0:8, 1:13].values.ravel()
    well_df["experiment_date"] = date_string

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
    root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
    experiment_date = "20231013"

    extract_frame_metadata(root=root, experiment_date=experiment_date)
