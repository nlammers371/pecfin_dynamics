import sys
sys.path.append('/home/nick/projects')
sys.path.append('/home/nick/projects/PointGPT')
sys.path.append('/home/nick/projects/PointGPT/segmentation')

import os
import numpy as np
import glob2 as glob
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.PointGPT_utils.dataset import FinDataset
from PointGPT.segmentation.models import pt_fin as pt
from src.utilities.functions import path_leaf
from sklearn.neighbors import KDTree
import scipy
from sklearn.cluster import KMeans

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def extract_point_cloud_features(root, point_cloud_size=8192, model_name='PointGPT_S', mdl_tag="", seg_pd_flag=False,
                                 cls_label=0, mdl_path=None, overwrite_flag=False, batch_size=16):

    seg_classes = {'tissue': [0, 1, 2, 3], 'tbx5a': np.arange(4, 25).tolist()}
    cat_list = list(seg_classes.keys())
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    if mdl_path is None:
        mdl_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/part_seg.pth"
        
    mdl_name = path_leaf(mdl_path).replace(".pth", "")

    outpath = os.path.join(root, "point_cloud_data", "point_features_" + mdl_name + mdl_tag, "")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    data_root = os.path.join(root, "point_cloud_data", "nucleus_point_clouds", "")

    # generate dataloader to load fin point clouds
    point_data = FinDataset(root=data_root, cls_label=cls_label, npoints=point_cloud_size, split='all',
                            outpath=outpath, overwrite=overwrite_flag)
    dataloader = DataLoader(point_data, batch_size=batch_size, shuffle=True)

    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'PointGPT_S':
        MODEL = pt
        if mdl_name == "part_seg":
            num_part = 50
            num_classes = 16
        elif mdl_name == "seg01_best_model":
            num_part = 25  # number of distinct parts to segment
            num_classes = 2  # number of kinds of object
        else:
            raise ValueError('Model type not supported')

        classifier = MODEL.get_model(cls_dim=num_part, num_class=num_classes, pd_flag=True,
                                     trans_dim=384, depth=12, drop_path_rate=0.1, num_heads=6,
                                     decoder_depth=4, group_size=32, num_group=128, prop_dim=1024, label_dim1=512,
                                     label_dim2=256, encoder_dims=384)
        classifier = classifier.cuda()
        # criterion = MODEL.get_loss().cuda()
        classifier.apply(inplace_relu)
    else:
        raise Exception('Model name must be "PointGPT_S"')

    # Load model checkpoint
    mdl_ckpt = torch.load(mdl_path, map_location=device)
    classifier.load_state_dict(mdl_ckpt["model_state_dict"])

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

        # pass points to model
        if len(raw_path) > 0:
            points = points.to(device)

            with torch.no_grad():
                f_level_0, x1, x2, x3, seg = classifier.forward(points, to_categorical(label, num_classes))
                # local_features, full_features, _, _ = classifier.extract_features(points, label)
            full_features = x2.detach().cpu().numpy()
            feature_cols = [f"feat_{f:04}" for f in range(full_features.shape[1])]
            if seg_pd_flag:
                cat = cat_list[label[0].detach().data.cpu().numpy()]
                full_class_pd = np.transpose(seg.detach().cpu().numpy(), (0, 2, 1))[:, seg_classes[cat], :]
                seg_cols = [f"class_prob_{f:04}" for f in range(full_class_pd.shape[1])]

            for p, path in enumerate(raw_path):

                # load DF
                point_df = pd.read_csv(path)
                feature_df = point_df.copy()

                # extract features
                features = np.squeeze(full_features[p]).T
                features_full = np.empty((feature_df.shape[0], features.shape[1]))
                features_full[:] = np.nan
                if seg_pd_flag:
                    seg_pd = np.squeeze(full_class_pd[p]).T
                    seg_pd_full = np.empty((feature_df.shape[0], seg_pd.shape[1]))
                    pd_array = np.empty((feature_df.shape[0],))

                # transfer feature info for points that were included in the network pass
                indices_raw = sample_indices[p, :]
                indices, ia = np.unique(indices_raw, return_index=True)
                features_to_use = features[ia]
                # feature_df.loc[indices, feature_cols] = features_to_use
                features_full[indices, :] = features_to_use
                if seg_pd_flag:
                    predictions_to_use = seg_pd[ia]
                    # feature_df.loc[indices, seg_cols] = predictions_to_use
                    seg_pd_full[indices, :] = predictions_to_use
                    # feature_df.loc[indices, "label_pd"] = np.argmax(predictions_to_use, axis=1)
                    pd_array[indices] = np.argmax(predictions_to_use, axis=1)
                # feature_df = feature_df.loc[indices]
                # use nn info to extend point features to all nuclei in the dataset
                indices_to_fill = np.where(np.isnan(features_full[:, 0]))[0]

                if len(indices_to_fill) > 0:
                    nn_k = 3
                    tree = KDTree(feature_df.loc[indices, ["Z", "Y", "X"]], leaf_size=2)
                    nearest_dist, nearest_ind = tree.query(feature_df.loc[indices_to_fill, ["Z", "Y", "X"]], k=nn_k + 1)

                    # get neighbors with non-missing values
                    nn_ind_vec = nearest_ind[:, 1:].ravel()

                    nn_feat_array = features_to_use[nn_ind_vec]
                    nn_feat_avg = (nn_feat_array[0::3] + nn_feat_array[1::3] + nn_feat_array[2::3]) / 3
                    # feature_df.loc[indices_to_fill, feature_cols] = nn_feat_avg
                    features_full[indices_to_fill, :] = nn_feat_avg
                    if seg_pd_flag:
                        nn_seg_array = predictions_to_use[nn_ind_vec]
                        nn_seg_avg = (nn_seg_array[0::3] + nn_seg_array[1::3] + nn_seg_array[2::3]) / 3
                        # feature_df.loc[indices_to_fill, seg_cols] = nn_seg_avg
                        seg_pd_full[indices_to_fill, :] = nn_seg_avg
                        # feature_df.loc[indices_to_fill, "label_pd"] = np.argmax(nn_seg_avg, axis=1)
                        pd_array[indices_to_fill] = np.argmax(nn_seg_avg, axis=1)

                # update DF
                feature_df.loc[:, feature_cols] = features_full
                feature_df = feature_df.copy()
                if seg_pd_flag:
                    feature_df.loc[:, seg_cols] = seg_pd_full
                    feature_df.loc[:, "label_pd"] = pd_array
                # save
                feature_df.to_csv(point_path_vec[p], index=False)

if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    extract_point_cloud_features(root, overwrite_flag=False)