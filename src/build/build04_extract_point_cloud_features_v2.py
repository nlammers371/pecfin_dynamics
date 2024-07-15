import os
import numpy as np
import glob2 as glob
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.PointGPT_utils.dataset import FinDataset
from src.PointGPT_utils.models import pt
from src.utilities.functions import path_leaf
from sklearn.neighbors import KDTree
import scipy
from sklearn.cluster import KMeans

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def extract_point_cloud_features(root, point_cloud_size=8192, model_name='PointGPT_S',
                                 mdl_path=None, fluo_channel=None, overwrite_flag=False, batch_size=16):

    if mdl_path is None:
        ckpt_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/part_seg.pth"

    # model_type = path_leaf(model_path)
    outpath = os.path.join(root, "point_cloud_data", "nucleus_point_features", "")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    data_root = os.path.join(root, "point_cloud_data", "nucleus_point_clouds", "")

    # generate dataloader to load fin point clouds
    point_data = FinDataset(root=data_root, npoints=point_cloud_size, split='all')
    dataloader = DataLoader(point_data, batch_size=batch_size, shuffle=True)

    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'PointGPT_S':
        MODEL = pt
        num_part = 50
        classifier = MODEL.get_model(num_part, trans_dim=384, depth=12, drop_path_rate=0.1, num_heads=6,
                                     decoder_depth=4, group_size=32, num_group=128, prop_dim=1024, label_dim1=512,
                                     label_dim2=256, encoder_dims=384)
        classifier = classifier.cuda()
        # criterion = MODEL.get_loss().cuda()
        classifier.apply(inplace_relu)
    else:
        raise Exception('Model name must be "PointGPT_S"')

    # Load model checkpoint
    mdl_ckpt = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(mdl_ckpt["model_state_dict"])
    # classifier.load_model_from_ckpt(ckpt_path)

    # apply to FOVs to generate training features
    for batch_i, batch in enumerate(tqdm(dataloader, "Extracting point features...")):

        points = batch[0]
        label = batch[1]
        target = batch[2]
        raw_path = batch[3][1]
        sample_indices = batch[4]

        points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()

        points = points.transpose(2, 1)

        point_path_vec = []
        new_indices = []
        for p, path in enumerate(raw_path):

            path_suffix = path_leaf(path).replace(".csv", "") + "_features.csv"
            point_path = os.path.join(outpath, path_suffix)

            if not os.path.isfile(point_path):
                new_indices.append(p)

            point_path_vec.append(point_path)

        # if not overwrite_flag:
        #     new_indices = np.asarray(new_indices)
        #     points = points[new_indices]
        #     raw_path = [raw_path[i] for i in new_indices]
        #     point_path_vec = [point_path_vec[i] for i in new_indices]
        #     sample_indices = sample_indices[new_indices]

        # pass points to model
        if len(raw_path) > 0:
            points = points.to(device)

            with torch.no_grad():
                local_features, full_features, _, _ = classifier.extract_features(points, label)

            feature_cols = [f"feat_{f:04}" for f in range(full_features.shape[1])]

            for p, path in enumerate(raw_path):

                if (p in new_indices) or overwrite_flag:
                    # load DF
                    point_df = pd.read_csv(path)
                    feature_df = point_df.copy()

                    # extract features
                    features = np.squeeze(full_features[p].detach().cpu().numpy()).T

                    # transfer feature info for points that were included in the network pass
                    indices_raw = sample_indices[p, :]
                    indices, ia = np.unique(indices_raw, return_index=True)
                    features_to_use = features[ia]
                    feature_df.loc[indices, feature_cols] = features_to_use
                    # feature_df = feature_df.loc[indices]
                    # use nn info to extend point features to all nuclei in the dataset
                    indices_to_fill = np.where(np.isnan(feature_df.loc[:, feature_cols[0]]))[0]

                    if len(indices_to_fill) > 0:
                        nn_k = 3
                        tree = KDTree(feature_df.loc[indices, ["Z", "Y", "X"]], leaf_size=2)
                        nearest_dist, nearest_ind = tree.query(feature_df.loc[indices_to_fill, ["Z", "Y", "X"]], k=nn_k + 1)

                        # get neighbors with non-missing values
                        nn_ind_vec = nearest_ind[:, 1:].ravel()

                        nn_feat_array = features_to_use[nn_ind_vec]
                        nn_feat_avg = (nn_feat_array[0::3] + nn_feat_array[1::3] + nn_feat_array[2::3]) / 3
                        feature_df.loc[indices_to_fill, feature_cols] = nn_feat_avg

                    # save
                    feature_df.to_csv(point_path_vec[p], index=False)

if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    extract_point_cloud_features(root, overwrite_flag=False)