from tqdm import tqdm
import glob2 as glob
import os
from typing import Any
from typing import Dict
from functions.utilities import path_leaf
import pandas as pd
import nd2

import numpy as np

def extract_frame_metadata(
    root: str,
    experiment_date: str,
    sheet_names = None
) -> Dict[str, Any]:

    if sheet_names is None:
        sheet_names = ["series_number_map", "genotype", "age_hpf"]

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


    # well_df["experiment_date"] = date_string

    # read the image data
    imObject = nd2.ND2File(nd2_path)
    im_raw_dask = imObject.to_dask()
    scale_vec = imObject.voxel_size()

    # read in label files
    label_dir = os.path.join(root, "built_data", "cellpose_output", )



    return {}

if __name__ == "__main__":

    # set path to CellPose model to use
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
    experiment_date = "20231213"

    extract_frame_metadata(root=root, experiment_date=experiment_date)
