import numpy as np
import pandas as pd
import open3d as o3d
from glob2 import glob
import os
from src.utilities.functions import path_leaf
from src.utilities.fin_class_def import FinData

# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

fin_object_path = os.path.join(root, "point_cloud_data", "fin_objects", "")
fin_object_list = sorted(glob(fin_object_path + "*.pkl"))

file_ind01 = 96
seg_type = "tissue_only_best_model_tissue"

fp01 = fin_object_list[file_ind01]
point_prefix01 = path_leaf(fp01).replace("_fin_object.pkl", "")
print(point_prefix01)

fin_data = FinData(data_root=root, name=point_prefix01, tissue_seg_model=seg_type)
fin_df = fin_data.full_point_data
fin_df = fin_df.loc[fin_df["fin_label_curr"]==1, :]
fin_points = fin_df[["X", "Y", "Z"]].to_numpy()

fin_axis_df = fin_data.axis_fin
fin_axes = fin_data.calculate_axis_array(fin_axis_df)
fin_points_pca = np.matmul(fin_points - np.mean(fin_points, axis=0), fin_axes.T)
fin_df.loc[:, ["XP", "YP", "ZP"]] = fin_points_pca

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(fin_points_pca)

import copy
k=10
# Downsample the point cloud for visualization purposes
pcd_down = pcd.voxel_down_sample(voxel_size=0.1)

# Re-estimate normals if necessary
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
)
pcd_down.orient_normals_consistent_tangent_plane(k)

# Create a list to hold geometry for visualization
geometries = [pcd_down]

# For each point, create a coordinate frame representing the local axes
for point, normal in zip(pcd_down.points, pcd_down.normals):
    # Create a coordinate frame at the point
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=point
    )
    # Align the z-axis of the frame with the normal
    rotation = frame.get_rotation_matrix_from_xyz((0, 0, 0))
    frame.rotate(rotation, center=point)
    geometries.append(frame)

# Visualize the point cloud with local frames
o3d.visualization.draw_geometries(geometries)