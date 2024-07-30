from src.build.build03_extract_point_clouds_v2 import extract_nucleus_stats, generate_fluorescence_labels, make_segmentation_training_folder
from src.build.build04_extract_point_cloud_features_v2 import extract_point_cloud_features

root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

# call feature extraction script for tbx5a and tissue classes
mdl_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/seg02_best_model.pth"
cls_label = 0  # tissue predictions
extract_point_cloud_features(root, overwrite_flag=True, mdl_path=mdl_path, cls_label=cls_label, seg_pd_flag=True, mdl_tag="_tissue")

# cls_label = 1  # tbx5a levels
# extract_point_cloud_features(root, overwrite_flag=True, mdl_path=mdl_path, cls_label=cls_label, seg_pd_flag=True, mdl_tag="_tbx5a")