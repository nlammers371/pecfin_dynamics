"""
Image segmentation via Cellpose library
"""
import logging
import glob2 as glob
import os
import time
# import pyclesperanto as cle
from typing import Any
from typing import Dict
from src.utilities.functions import path_leaf
import zarr
from tqdm import tqdm

# logging = logging.getlogging(__name__)
logging.basicConfig(level=logging.NOTSET)
# __OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__



def correct_metadata(
    *,
    # Fractal arguments
    root: str,
    experiment_date: str,
) -> Dict[str, Any]:


    # path to zarr files
    data_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')

    # get list of images
    image_list = sorted(glob.glob(data_directory + "*.zarr"))

    for well_index in tqdm(range(0, len(image_list)), "Correcting metadata..."):

        zarr_path = image_list[well_index]
        im_name = path_leaf(zarr_path)
        print("processing " + im_name)
        # read the image data
        data_tzyx = zarr.open(zarr_path, mode="a")
        data_tzyx.attrs["nuclear_channel"] = 1
        data_tzyx.attrs["channel_names"] = ['tbx5a-StayGold', 'H2B-tdTom']



    return {}

if __name__ == "__main__":
    # s0rt some hyperparameters
    overwrite = False
    xy_ds_factor = 1
    cell_diameter = 10
    cellprob_threshold = 0.0
    pixel_res_raw = [2, 0.55, 0.55]

    # set path to CellPose model to use
    pretrained_model = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/cellpose_training/20240424_tdTom/log/models/log-v3"

    # set read/write paths
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date_vec = ["20240424", "20240425"]

    for experiment_date in experiment_date_vec:
        correct_metadata(root=root, experiment_date=experiment_date)
