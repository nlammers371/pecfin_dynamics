from src.build._Archive.build01_segment_nuclei_zarr import cellpose_segmentation

overwrite = False
cell_diameter = 10
cellprob_threshold = 0.0

# set path to CellPose model to use
# pretrained_model0 = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/cellpose_training/standard_models/tdTom-bright-log-v5"
# pretrained_model1 = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/cellpose_training/standard_models/tdTom-dim-log-v3"
pretrained_model0 = "/net/trapnell/vol1/home/nlammers/projects/data/pecfin_dynamics/built_data/cellpose_training/standard_models/tdTom-bright-log-v5"
pretrained_model1 = "/net/trapnell/vol1/home/nlammers/projects/projects/data/pecfin_dynamics/built_data/cellpose_training/standard_models/tdTom-dim-log-v3"

# set read/write paths
# root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
root= "/net/trapnell/vol1/home/nlammers/projects/data/pecfin_dynamics/"
experiment_date_vec = ["20240619", "20240620"] #["20240223", "20240424", "20240425", "20240619", "20240620"]
pretrained_model_vec = [pretrained_model0, pretrained_model0] #[pretrained_model0, pretrained_model1, pretrained_model1, pretrained_model0, pretrained_model0]
nuclear_channel_vec = [0, 0] #[0, 1, 1, 0, 0]
for e, experiment_date in enumerate(experiment_date_vec):
    pretrained_model = pretrained_model_vec[e]
    nuclear_channel = nuclear_channel_vec[e]
    cellpose_segmentation(root=root, experiment_date=experiment_date,
                          xy_ds_factor=1, cell_diameter=cell_diameter, nuclear_channel=nuclear_channel,
                          cellprob_threshold=cellprob_threshold, pretrained_model=pretrained_model, overwrite=overwrite)