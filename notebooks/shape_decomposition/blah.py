from src.utilities.fin_shape_utils import fit_fin_hull, upsample_fin_point_cloud
from src.utilities.fin_class_def import FinData
from src.utilities.functions import path_leaf
import pandas as pd
import os
import numpy as np
import glob2 as glob

# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
fin_object_path = os.path.join(root, "point_cloud_data", "fin_objects", "")
fin_object_list = sorted(glob.glob(fin_object_path + "*.pkl"))

file_ind01 = 146
seg_type = "tissue_only_best_model_tissue"
fp01 = fin_object_list[file_ind01]
point_prefix01 = path_leaf(fp01).replace("_fin_object.pkl", "")
print(point_prefix01)

fin_object = FinData(data_root=root, name=point_prefix01, tissue_seg_model=seg_type)

fin_df_upsamp = upsample_fin_point_cloud(fin_object, sample_res_um=0.5, root=root, points_per_nucleus=50)
fin_df_upsamp.head()