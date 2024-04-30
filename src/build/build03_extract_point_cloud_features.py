import os
import numpy as np
import glob2 as glob
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.utilities.data_utilities import PointData
from src.point_net.point_net import PointNetSegHead
from skimage.measure import regionprops
import zarr
from src.utilities.functions import path_leaf

def labels_to_point_cloud(root, experiment_date, seg_model, well_num, time_int, mask, scale_vec, overwrite=False):

    # check if file exists
    point_prefix = experiment_date + f"_well{well_num:04}" + f"_time{time_int:04}"
    point_path = os.path.join(root, "built_data", "point_clouds", experiment_date, "")
    if not os.path.isdir(point_path):
        os.makedirs(point_path)

    if (not os.path.isfile((point_path + point_prefix + "_centroids.csv"))) | overwrite:
        # use regionprops to get centroids for each nucleus label
        regions = regionprops(mask)

        if len(regions) > 0:
            centroid_array = np.asarray([rg["Centroid"] for rg in regions])
            centroid_array = np.multiply(centroid_array, scale_vec)

            # convert to dataframe
            point_df = pd.DataFrame(centroid_array, columns=["Z", "Y", "X"])

            # add metadata
            point_df["experiment_date"] = experiment_date
            point_df["seg_model"] = seg_model
            point_df["well_num"] = well_num
            point_df["time_int"] = time_int
            point_df["fin_curation_flag"] = False

            # save
            point_df.to_csv(point_path + point_prefix + "_centroids.csv", index=False)

        else:
            point_df = []
    else:
        point_df = pd.read_csv(point_path + point_prefix + "_centroids.csv")

    return point_df

def point_cloud_wrapper(root, experiment_date, seg_model, scale_vec, overwrite=False, suffix_string="_stitched.zarr"):

    # get list of zarr mask files
    mask_dir = os.path.join(root, "built_data", "stitched_labels", seg_model, experiment_date, "")
    mask_file_list = sorted(glob.glob(mask_dir + "*" + suffix_string))

    for mask_file in tqdm(mask_file_list):

        mask_zarr = zarr.open(mask_file, mode="r")
        n_time_points = mask_zarr.shape[0]

        mask_name = path_leaf(mask_file)
        ind = mask_name.find("well")
        well_num = int(mask_name[ind + 4:ind + 8])

        for time_int in tqdm(range(n_time_points)):
            _ = labels_to_point_cloud(root, experiment_date, seg_model, well_num, time_int, mask_zarr[time_int], scale_vec, overwrite)


def extract_point_cloud_features(root, model_root, model_name, experiment_date, overwrite_flag=False):

    outpath = os.path.join(root, "built_data", "processed_point_clouds", experiment_date, "")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)


    model_path = os.path.join(model_root, model_name)
    # data_root = os.path.join(root,  "built_data", "point_clouds", "_Archive", "")
    data_root = os.path.join(root, "built_data", "point_clouds", "")

    # feature selection hyperparameters
    NUM_TEST_POINTS = 4*4096
    BATCH_SIZE = 16
    NUM_CLASSES = 14
    n_local = 64

    # generate dataloader to load fin point clouds
    point_data = PointData(data_root, split='test', npoints=NUM_TEST_POINTS)
    dataloader = DataLoader(point_data, batch_size=BATCH_SIZE, shuffle=True)

    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load pretrained model
    model = PointNetSegHead(num_points=NUM_TEST_POINTS, m=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # apply to FOVs to generate training features
    print("Extracting point features...")
    for batch_i, batch in enumerate(tqdm(dataloader)):
        # extract data
        points = batch["data"]
        points = torch.transpose(points, 1, 2)
        raw_path = batch["path"]
        sample_indices = batch["point_indices"]

        point_path_vec = []
        global_path_vec = []
        new_indices = []
        for p, path in enumerate(raw_path):
            path_suffix = path.replace(data_root, "").split("\\")
            date_folder = os.path.join(outpath, path_suffix[0])
            if not os.path.isdir(date_folder):
                os.makedirs(date_folder)

            point_path = os.path.join(date_folder, path_suffix[1])
            global_name = path_suffix[1].replace(".csv", "_global.csv")
            global_path = os.path.join(date_folder, global_name)
            if not os.path.isfile(point_path):
                new_indices.append(p)

            point_path_vec.append(point_path)
            global_path_vec.append(global_path)

        if not overwrite_flag:
            new_indices = np.asarray(new_indices)
            points = points[new_indices]
            raw_path = [raw_path[i] for i in new_indices]
            point_path_vec = [point_path_vec[i] for i in new_indices]
            global_path_vec = [global_path_vec[i] for i in new_indices]
            sample_indices = sample_indices[new_indices]

        # pass points to model
        if len(raw_path) > 0:
            points = points.to(device)
            backbone = model.backbone
            pointfeat, _, _ = backbone(points)

            # now, load point data frames and add point features
            col_names_local = [f"feat_loc{f:04}" for f in range(n_local)]
            col_names_global = [f"feat_glob{f:04}" for f in range(pointfeat.shape[1] - n_local)]
            # col_names = col_names_local + col_names_global

            for p, path in enumerate(raw_path):
                # load DF
                point_df = pd.read_csv(path)

                # extract features
                features = np.asarray(np.squeeze(pointfeat[p, :, :]).detach().cpu().T)

                indices_raw = sample_indices[p, :]
                indices, ia = np.unique(indices_raw, return_index=True)
                # assign features to dataframe
                # pt = np.squeeze(points_raw[p, :, :])
                point_df.loc[indices, col_names_local] = features[ia, :n_local]

                # make global df
                global_df = pd.DataFrame(np.reshape(features[0, n_local:], (1, len(col_names_global))), columns=col_names_global)

                point_df.to_csv(point_path_vec[p])
                global_df.to_csv(global_path_vec[p])

if __name__ == '__main__':
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\"
    model_root = "C:\\Users\\nlammers\\Projects\\pecfin_dynamics\\fin_morphodynamics\\src\\point_net\\trained_models\\"
    model_name = "seg_focal_dice_iou_rot\\seg_model_89.pth"

    experiment_date = "20240223"
    seg_model = "log-v5"
    scale_vec = np.asarray([2.0, 0.55, 0.55])
    # build point cloud files
    # point_cloud_wrapper(root, experiment_date, seg_model, scale_vec, overwrite=False)

    extract_point_cloud_features(root, model_root, model_name, experiment_date=experiment_date)