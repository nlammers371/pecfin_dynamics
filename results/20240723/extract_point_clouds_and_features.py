from src.build.build03_extract_point_clouds_v2 import extract_nucleus_stats
from src.build.build04_extract_point_cloud_features_v2 import extract_point_cloud_features

root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

experiment_date_vec = ["20240424", "20240425"]
seg_model_vec = ["tdTom-dim-log-v3", "tdTom-dim-log-v3"]  #["log-v3", "log-v3", "log-v5"]
# build point cloud files
for e, experiment_date in enumerate(experiment_date_vec):
    extract_nucleus_stats(root, experiment_date, seg_model_vec[e], overwrite_flag=False)

# call feature extraction script
extract_point_cloud_features(root, overwrite_flag=False)