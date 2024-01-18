from tqdm import tqdm
import glob2 as glob
import os
from typing import Any
from typing import Dict
from functions.utilities import path_leaf
import pandas as pd
import nd2
import openpyxl
import numpy as np

def extract_frame_metadata(
    root: str,
    experiment_date: str,
    sheet_names = None
) -> Dict[str, Any]:

    if sheet_names is None:
        sheet_names = ["series_number_map", "genotype_map", "age_hpf"]

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

    ####################
    # Process information from plate map
    well_coord_list = []
    for row in row_letters:
        for col in col_nums:
            well_coord_list.append(row + f"{col:02}")

    well_coord_list = np.asarray(well_coord_list)
    print("processing " + im_name)

    xl_temp = pd.ExcelFile(plate_directory)
    # extract nd2 series info
    series_vec_raw = xl_temp.parse(sheet_names[0]).iloc[0:8, 1:13].values.ravel()
    nn_indices = np.where(~np.isnan(series_vec_raw))[0]
    series_vec = series_vec_raw[nn_indices]
    # extract staging info
    start_age_hpf = xl_temp.parse(sheet_names[2]).iloc[0:8, 1:13].values.ravel()
    start_age_hpf = start_age_hpf[nn_indices]
    # extract genotype info
    genotype_vec = xl_temp.parse(sheet_names[1]).iloc[0:8, 1:13].values.ravel()
    genotype_vec = genotype_vec[nn_indices]

    # get list of well ID coordinates
    well_coord_list = well_coord_list[nn_indices]

    plate_df = pd.DataFrame(series_vec[:, np.newaxis], columns=["nd2_series"])
    plate_df["well_id"] = well_coord_list
    plate_df["genotype"] = genotype_vec
    plate_df["start_age_hpf"] = start_age_hpf

    ####################
    # Process extract information from nd2 metadata
    ####################
    # xyz pixel resolution; xyz stage position; time stamp
    # read the image data
    imObject = nd2.ND2File(nd2_path)
    im_raw_dask = imObject.to_dask()
    im_shape = im_raw_dask.shape
    n_z_slices = im_shape[2]
    n_time_points = im_shape[0]
    n_wells = im_shape[1]

    scale_vec = imObject.voxel_size()

    # extract frame times
    n_frames_total = imObject.frame_metadata(0).contents.frameCount
    frame_time_vec = [imObject.frame_metadata(i).channels[0].time.relativeTimeMs / 1000 for i in
                      range(0, n_frames_total, im_shape[2])]

    # check for common nd2 artifact where time stamps jump midway through
    dt_frame_approx = (imObject.frame_metadata(n_z_slices).channels[0].time.relativeTimeMs -
                       imObject.frame_metadata(0).channels[0].time.relativeTimeMs) / 1000
    jump_ind = np.where(np.diff(frame_time_vec) > 2 * dt_frame_approx)[0]  # typically it is multiple orders of magnitude to large
    if len(jump_ind) > 0:
        jump_ind = jump_ind[0]
        # prior to this point we will just use the time stamps. We will extrapolate to get subsequent time points
        nf = jump_ind - 1 - int(jump_ind / 2)
        dt_frame_est = (frame_time_vec[jump_ind - 1] - frame_time_vec[int(jump_ind / 2)]) / nf
        base_time = frame_time_vec[jump_ind - 1]
        for f in range(jump_ind, len(frame_time_vec)):
            frame_time_vec[f] = base_time + dt_frame_est * (f - jump_ind)
    frame_time_vec = np.asarray(frame_time_vec)

    # get well positions
    stage_zyx_array = np.empty((n_wells * n_time_points, 3))
    for t in range(n_time_points):
        for w in range(n_wells):
            base_ind = t * n_wells + w
            slice_ind = base_ind * n_z_slices

            stage_zyx_array[base_ind, :] = np.asarray(
                imObject.frame_metadata(slice_ind).channels[0].position.stagePositionUm)[::-1]

    ###################
    # Pull it together into dataframe
    ###################
    well_df = pd.DataFrame(np.tile(range(1, n_wells+1), n_time_points)[:, np.newaxis], columns=["nd2_series"])
    well_df["time_index"] = np.repeat(range(n_time_points), n_wells)

    # join on plate info using series id
    well_df = well_df.merge(plate_df, on="nd2_series", how="left")

    # add additional info
    well_df["time"] = frame_time_vec
    well_df["stage_z_um"] = stage_zyx_array[:, 0]
    well_df["stage_y_um"] = stage_zyx_array[:, 1]
    well_df["stage_x_um"] = stage_zyx_array[:, 2]

    well_df["x_res_um"] = scale_vec[0]
    well_df["y_res_um"] = scale_vec[1]
    well_df["z_res_um"] = scale_vec[2]

    ################
    # Finally, add curation info
    curation_path = os.path.join(root, "metadata", "curation", experiment_date + "_curation_info.xlsx")
    curation_xl = pd.ExcelFile(curation_path)
    curation_df = curation_xl.parse(curation_xl.sheet_names[0])
    # read in label files
    label_dir = os.path.join(root, "built_data", "cellpose_output", )



    return {}

if __name__ == "__main__":

    # set path to CellPose model to use
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
    experiment_date = "20231214"

    extract_frame_metadata(root=root, experiment_date=experiment_date)
