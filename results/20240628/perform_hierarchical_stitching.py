from src.build.build02_stitch_nuclear_masks import stitch_cellpose_labels

# set read/write paths
# root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
root= "/net/trapnell/vol1/home/nlammers/projects/data/pecfin_dynamics/"
experiment_date_vec = ["20240223"]
pretrained_model_vec = ["tdTom-bright-log-v5"] #[pretrained_model0, pretrained_model1, pretrained_model1, pretrained_model0, pretrained_model0]
overwrite = True

for e, experiment_date in enumerate(experiment_date_vec):

    model_name = pretrained_model_vec[e]

    stitch_cellpose_labels(root=root, model_name=model_name, experiment_date=experiment_date, overwrite=overwrite)