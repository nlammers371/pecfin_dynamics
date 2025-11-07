from src.build._Archive.build02_stitch_nuclear_masks import stitch_cellpose_labels
import numpy as np

# set read/write paths
# root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
root = "Y:\\data\\pecfin_dynamics\\"
experiment_date_vec = ["amullen"]#, "20240712_01", "20240712_02"]
pretrained_model = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/cellpose_training/general/models/DAPI-Pro-7"
pretrained_model_vec = [pretrained_model]  #[pretrained_model0, pretrained_model1, pretrained_model1, pretrained_model0, pretrained_model0]
overwrite = True
prob_thresh_range = np.arange(-9, 10, 3)
well_range = None
seg_res = 0.65

for e, experiment_date in enumerate(experiment_date_vec):

    model_name = pretrained_model_vec[e]

    stitch_cellpose_labels(root=root, model_name=model_name, prob_thresh_range=prob_thresh_range, well_range=well_range,
                                      experiment_date=experiment_date, overwrite=overwrite, seg_res=seg_res)