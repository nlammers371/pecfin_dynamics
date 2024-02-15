# dataset
# ROOT = r'C:\Users\itber\Documents\datasets\S3DIS\Stanford3dDataset_v1.2_Reduced_Parti\tioned_Aligned_Version_1m'
import os
from torch.utils.data import DataLoader
from fin_morphodynamics.src.functions.data_utilities import PointData



root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\"
data_root = os.path.join(root, "built_data\\point_clouds\\")

# feature selection hyperparameters
NUM_TRAIN_POINTS = 4096 # train/valid points
NUM_TEST_POINTS = 4096
BATCH_SIZE = 16

# get datasets
# point_data = PointData(ROOT, npoints=NUM_TRAIN_POINTS, r_prob=0.25)
# valid_data = PointData(ROOT, npoints=NUM_TRAIN_POINTS, r_prob=0.)
point_data = PointData(data_root, split='test', npoints=NUM_TEST_POINTS)

# get dataloaders
dataloader = DataLoader(point_data, batch_size=BATCH_SIZE, shuffle=True)

points, targets = next(iter(dataloader))
print("check")