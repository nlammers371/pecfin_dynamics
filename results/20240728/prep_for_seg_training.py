from src.build.build03_extract_point_clouds_v2 import extract_nucleus_stats, generate_fluorescence_labels, make_segmentation_training_folder


root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

# normalize and discretize fluorescence labels
fluo_df_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/point_cloud_data/tbx5a_segmentation/"
fluo_var = "tbx5a-StayGold_mean_nn"
# generate_fluorescence_labels(fluo_df_path, fluo_var, nbins=7)

# make segmentation training folder
make_segmentation_training_folder(root)