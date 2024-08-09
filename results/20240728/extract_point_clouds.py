from src.build.build03_extract_point_clouds_v2 import extract_nucleus_stats, generate_fluorescence_labels, make_segmentation_training_folder
from src.build.build04_extract_point_cloud_features_v2 import extract_point_cloud_features

root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

experiment_date_vec = ["20240424", "20240425"] #["20240711_01", "20240712_01", "20240712_02"]
seg_model_vec = ["tdTom-dim-log-v3", "tdTom-dim-log-v3"] #["tdTom-bright-log-v5", "tdTom-bright-log-v5", "tdTom-bright-log-v5", "tdTom-bright-log-v5"]  #["log-v3", "log-v3", "log-v5"]

# build point cloud files
for e, experiment_date in enumerate(experiment_date_vec):
    extract_nucleus_stats(root, experiment_date, seg_model_vec[e], overwrite_flag=False)

# mdl_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/seg01_best_model.pth"
# cls_label = 0  # tissue predictions
# extract_point_cloud_features(root, overwrite_flag=False, mdl_path=mdl_path, cls_label=cls_label, seg_pd_flag=True, mdl_tag="_tissue")