from src.build.build03_extract_point_clouds_v2 import make_vae_training_data


root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
seg_model = "tissue_only_best_model_tissue"

# make segmentation training folder
make_vae_training_data(root=root, seg_model=seg_model, out_suffix="_20240825")