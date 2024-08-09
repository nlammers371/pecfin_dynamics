import sys
sys.path.append('/home/nick/projects')
# from PointGPT import segmentation
from PointGPT.segmentation.main_fin_seg import parse_args, main_training

if __name__ == "__main__":
    # pass run-specific args
    # args = parse_args()
    # args.ckpts = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/pretraining/ckpt-epoch-300.pth"
    # args.root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/point_cloud_data/segmentation_training_tissue_only/"
    # args.model_dir = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/segmentation/models/"
    # args.seg_classes = {'tissue': [0, 1, 2, 3]}
    custom_args = ['--ckpts', "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/pretraining/ckpt-epoch-300.pth",
                   '--root', "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/point_cloud_data/segmentation_training_tissue_only/",
                   '--model_dir', "/home/nick/projects/PointGPT/segmentation/models/",
                   '--save_dir', "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/tissue_only/"]
    seg_classes = {'tissue': [0, 1, 2, 3]}
    args = parse_args(custom_args)
    # call main training function
    main_training(args, seg_classes=seg_classes)