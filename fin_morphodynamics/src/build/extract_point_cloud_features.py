import os
import re
from glob import glob
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from fin_morphodynamics.src.functions.data_utilities import PointData
from fin_morphodynamics.src.point_net.point_net import PointNetSegHead


def extract_point_cloud_features(root, model_root, model_name, overwrite_flag=False):

    outpath = os.path.join(root, "built_data", "processed_point_clouds", "")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)


    model_path = os.path.join(model_root, model_name)
    data_root = os.path.join(root, "built_data\\point_clouds\\")

    # feature selection hyperparameters
    NUM_TEST_POINTS = 4096
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

        # points_raw = batch["raw_data"]

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
                point_df = pd.read_csv(path, index_col=0)

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

    extract_point_cloud_features(root, model_root, model_name)