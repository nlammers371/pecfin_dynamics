import os
import numpy as np
import glob2 as glob
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.utilities.data_utilities import PointDataReg
from src.point_net_nl.point_net_flex import PointNetRegHead
from skimage.measure import regionprops
import zarr
from src.utilities.functions import path_leaf
from sklearn.neighbors import KDTree
import scipy
from sklearn.cluster import KMeans

def extract_point_cloud_features(root, model_path, experiment_date, point_features=None,
                                 fluo_channel=None, overwrite_flag=False, point_cloud_size=4096, batch_size=16):

    model_type = path_leaf(model_path)
    outpath = os.path.join(root, "built_data", "nucleus_data", "processed_point_clouds", model_type, experiment_date, "")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    if point_features is None:
        point_features = ["size", "eccentricity", "pZ_nn", "pY_nn", "pX_nn"]

    # model_path = os.path.join(model_root, model_name)
    # data_root = os.path.join(root,  "built_data", "point_clouds", "_Archive", "")
    data_root = os.path.join(root, "built_data", "nucleus_data", "point_clouds", "")

    # generate dataloader to load fin point clouds
    point_data = PointDataReg(data_root, split='test', training_dates=[experiment_date], point_features=point_features,
                                                                    npoints=point_cloud_size, fluo_channel=fluo_channel)
    dataloader = DataLoader(point_data, batch_size=batch_size, shuffle=True)

    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load pretrained model
    model = PointNetRegHead(num_points=point_cloud_size, num_point_features=len(point_features)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # apply to FOVs to generate training features
    for batch_i, batch in enumerate(tqdm(dataloader, "Extracting point features...")):
        # extract data
        points = batch["data"]
        points = torch.transpose(points, 1, 2)
        raw_path = batch["path"]
        sample_indices = batch["point_indices"]

        point_path_vec = []
        new_indices = []
        for p, path in enumerate(raw_path):


            path_suffix = path_leaf(path)
            point_path = os.path.join(outpath, path_suffix)

            if not os.path.isfile(point_path):
                new_indices.append(p)

            point_path_vec.append(point_path)

        if not overwrite_flag:
            new_indices = np.asarray(new_indices)
            points = points[new_indices]
            raw_path = [raw_path[i] for i in new_indices]
            point_path_vec = [point_path_vec[i] for i in new_indices]
            sample_indices = sample_indices[new_indices]

        # pass points to model
        if len(raw_path) > 0:
            points = points.to(device)

            _, _, _, pointfeat = model(points)

            # now, load point data frames and add point features
            col_names = [f"feat{f:04}" for f in range(pointfeat.shape[1])]
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
                point_df.loc[indices, col_names] = features[ia, :]
                if np.any(np.isnan(point_df.loc[:, col_names])):
                    print("wtf")
                point_df.to_csv(point_path_vec[p], index=False)

if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    model_dir = os.path.join(root, "built_data", "nucleus_data", "point_models_pos", "*")
    model_list = sorted(glob.glob(model_dir))
    model_path = os.path.join(model_list[-1], "*.pth")  # take most recent
    print(model_path)
    train_iter_list = sorted(glob.glob(model_path))
    train_iter_path = train_iter_list[-1]
    print(train_iter_path)

    experiment_dates = ["20240424", "20240425", "20240223"]

    for date in experiment_dates:
        extract_point_cloud_features(root, train_iter_path, experiment_date=date, overwrite_flag=True, point_features=[])