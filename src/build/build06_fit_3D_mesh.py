import flowshape as fs
import igl
import numpy as np
import meshplot as mp
import os
from src.utilities.fin_shape_utils import fit_fin_mesh, upsample_fin_point_cloud, plot_mesh
from src.utilities.fin_class_def import FinData
from src.utilities.functions import path_leaf
import glob2 as glob
from sklearn.metrics import pairwise_distances
import open3d as o3d
import trimesh
from tqdm import tqdm


def smooth_mesh_base(fin_mesh, fin_df, yolk_xyz, surf_center_o, fin_axes, base_axes, n_smooth_iters=35,
                     weight_hm=-2, weight_temperature=5):

    yolk_xyz_o = np.matmul(yolk_xyz - np.mean(fin_df[["X", "Y", "Z"]].to_numpy(), axis=0), fin_axes.T)
    surf_points = yolk_xyz_o.copy()  # [keep_flag_surf, :]
    surf_points_ro = np.dot(surf_points - surf_center_o, base_axes.T)

    # Calculate distance from  yolk surface to ellipsoid points
    fin_mesh_points = np.asarray(fin_mesh.vertices)
    dist_array = pairwise_distances(fin_mesh_points, surf_points_ro)
    yolk_dist = np.min(dist_array, axis=1)
    min_i = np.argmin(dist_array, axis=1)
    yolk_signs = np.sign(fin_mesh_points[:, 2] - surf_points_ro[min_i, 2])
    dist_metric = np.multiply(yolk_dist, yolk_signs)

    # use distances to calculate smoothing weights
    sm_weight_vec = np.divide(np.exp(-(dist_metric - weight_hm) / weight_temperature), 1 + np.exp(-(dist_metric - weight_hm) / weight_temperature))
    sm_weight_vec[sm_weight_vec < 0.05] = 0

    smoothed_mesh = fin_mesh.copy()

    for _ in tqdm(range(n_smooth_iters), "Smoothing mesh base..."):
        sm_temp = smoothed_mesh.copy()
        trimesh.smoothing.filter_laplacian(sm_temp, iterations=1)
        deltas = sm_temp.vertices - smoothed_mesh.vertices
        deltas_w = np.multiply(sm_weight_vec[:, np.newaxis], deltas)

        smoothed_mesh.vertices += deltas_w

    return smoothed_mesh


def get_yolk_distance(fin_object, yolk_dist_thresh=-10):

    # get fin points
    full_df = fin_object.full_point_data
    fin_df = full_df.loc[full_df["fin_label_curr"] == 1, :].reset_index(drop=True)

    if fin_df.shape[0] > 0:
        # orient to biological axes
        fin_axis_df = fin_object.axis_fin
        fin_axes = fin_object.calculate_axis_array(fin_axis_df)

        # Use simple numerical procedure to calculate distance of each fin point to the yolk
        fin_points = fin_df[["X", "Y", "Z"]].to_numpy()
        shift_ref_vec = np.mean(fin_points, axis=0)

        fin_points_pca = np.matmul(fin_points - shift_ref_vec, fin_axes.T)
        fin_df.loc[:, ["XP", "YP", "ZP"]] = fin_points_pca

        params = fin_object.yolk_surf_params

        x_min, y_min = fin_points[:, 0].min(), fin_points[:, 1].min()
        x_max, y_max = fin_points[:, 0].max(), fin_points[:, 1].max()

        # Create a mesh grid for x and y values
        x_vals = np.linspace(x_min, x_max, 150)
        y_vals = np.linspace(y_min, y_max, 150)
        X, Y = np.meshgrid(x_vals, y_vals)

        yolk_xyz = np.reshape(fin_object.polyval2d(np.c_[X.ravel(), Y.ravel()], params).ravel(), (-1, 3))

        dist_array = pairwise_distances(fin_points, yolk_xyz)
        yolk_dist = np.min(dist_array, axis=1)
        min_i = np.argmin(dist_array, axis=1)
        yolk_signs = np.sign(fin_points[:, 2] - yolk_xyz[min_i, 2])
        yolk_dist = -np.multiply(yolk_dist, yolk_signs)

        fin_df["yolk_dist"] = yolk_dist

        # filter out nuclei too far below yolk surface
        fin_df.reset_index(inplace=True, drop=True)
        dist_filter = (fin_df["yolk_dist"] >= yolk_dist_thresh).to_numpy()
        nuclei_to_keep = fin_df.loc[dist_filter, "nucleus_id"].to_numpy()

        return fin_df, yolk_xyz, fin_axes, nuclei_to_keep

    else:
        return None, None, None, None

def get_base_axes(fin_object, fin_df, yolk_xyz, fin_axes, yolk_thresh=5):

    # base_fin_points = fin_df.loc[np.abs(fin_df["yolk_dist"]) <= yolk_thresh, ["XP", "YP", "ZP"]].to_numpy()
    base_fin_points_raw = fin_df.loc[np.abs(fin_df["yolk_dist"]) <= yolk_thresh, ["X", "Y", "Z"]].to_numpy()

    # calculate axis dims. Main one we care about is the AP axis ("YP")
    # axis_len_vec = np.max(base_fin_points, axis=0) - np.min(base_fin_points, axis=0)

    # find centroid
    ref10 = np.percentile(base_fin_points_raw, 10, axis=0)
    ref90 = np.percentile(base_fin_points_raw, 90, axis=0)
    point_center = (ref10 + ref90) / 2  # np.mean(base_fin_points_raw, axis=0)
    surf_center_i = np.argmin(np.sqrt(np.sum((yolk_xyz - point_center) ** 2, axis=1)))
    surf_center = yolk_xyz[surf_center_i, :]  # this is the one we will use

    # define a local DV direction that is the cross product of the surface normal and the AP axis
    surf_normal_raw, _ = fin_object.calculate_tangent_plane(fin_object.yolk_surf_params, surf_center)
    if surf_normal_raw[2] > 0:
        surf_normal_raw = -surf_normal_raw

    # convert the normal vector to the biological axis space
    surf_normal = np.matmul(np.reshape(surf_normal_raw, (1, 3)), fin_axes.T)[0]
    surf_normal = surf_normal / np.linalg.norm(surf_normal)

    # calculate local DV
    dv_vec_base = np.cross(surf_normal, np.asarray([0, 1, 0]))
    dv_vec_base = dv_vec_base / np.linalg.norm(dv_vec_base)

    # define a third vector that is orthogonal to AP and local (base) DV)
    surf_vec_rel = np.cross([0, 1, 0], dv_vec_base)
    surf_vec_rel = surf_vec_rel / np.linalg.norm(surf_vec_rel)

    base_axes = np.stack([dv_vec_base, np.asarray([0, 1, 0]), surf_vec_rel], axis=1)

    # shift centerpoint into the oriented frame of reference
    shift_ref_vec0 = np.mean(fin_df[["X", "Y", "Z"]].to_numpy(), axis=0)
    shift_ref_vec1 = np.mean(fin_df[["XP", "YP", "ZP"]].to_numpy(), axis=0)
    surf_center_o = np.matmul(surf_center - shift_ref_vec0, fin_axes.T)
    surf_center_b = np.matmul(surf_center_o - shift_ref_vec1, base_axes.T)

    return base_axes, surf_center_b, surf_center_o


def fin_mesh_wrapper(root, overwrite_flag=False, sampling_res=0.5, seg_type="tissue_only_best_model_tissue"):

    # get list of fin objects
    fin_object_path = os.path.join(root, "point_cloud_data", "fin_objects", "")
    fin_object_list = sorted(glob.glob(fin_object_path + "*.pkl"))

    # make write directory
    write_dir = os.path.join(root, "point_cloud_data", "processed_fin_data", "")
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    #################
    # load fin object
    wt_vec = []
    for file_ind, fp in enumerate(tqdm(fin_object_list)):

        point_prefix = path_leaf(fp).replace("_fin_object.pkl", "")
        print(f"processing {point_prefix}...")

        fin_object = FinData(data_root=root, name=point_prefix, tissue_seg_model=seg_type)

        test_path = os.path.join(write_dir, point_prefix + "_smoothed_fin_mesh.obj")
        if overwrite_flag or (not os.path.isfile(test_path)):

            ############
            # calculate ditances to yolk (and load fin_df)
            fin_df, yolk_xyz, fin_axes, nuclei_to_keep = get_yolk_distance(fin_object)

            if fin_df is not None:
                ############
                # find base centerpoint and orientation
                base_axes, surf_center_b, surf_center_o = get_base_axes(fin_object, fin_df, yolk_xyz, fin_axes)

                ################
                # Upsample fin points
                fin_df_upsamp = upsample_fin_point_cloud(fin_object, sample_res_um=sampling_res, root=root, points_per_nucleus=100)

                ################
                # Orient and resample fin points. FIT MESH
                # shift fin points
                fin_points_b = np.matmul(fin_df_upsamp[["XP", "YP", "ZP"]].to_numpy() - surf_center_o, base_axes.T)
                fin_df_upsamp[["XB", "YB", "ZB"]] = fin_points_b

                # get raw points
                nc_vec_us = fin_df_upsamp.loc[:, "nucleus_id"].to_numpy().astype(np.uint16)
                keep_filter = np.isin(nc_vec_us, nuclei_to_keep)
                fin_points = fin_df_upsamp.loc[keep_filter, ["XB", "YB", "ZB"]].to_numpy()

                # convert to point cloud format
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(fin_points)

                # resample points to be more spatially uniform
                sampled_points = pcd.voxel_down_sample(voxel_size=sampling_res)

                # fit fin mesh
                print("Fitting fin mesh...")
                mesh_alpha = 25
                watertight_flag = False
                fin_points_u = np.asarray(sampled_points.points)
                while not watertight_flag:
                    fin_mesh, raw_mesh, watertight_flag = fit_fin_mesh(fin_points_u, alpha=mesh_alpha)

                    mesh_alpha -= 1 # decrement
                    if mesh_alpha <= 20:
                        break

                ##############
                # Apply additional smoothing to mesh regions near the fin base
                smoothed_mesh = smooth_mesh_base(fin_mesh, fin_df, yolk_xyz, surf_center_o, fin_axes, base_axes)

                # save
                fin_df.to_csv(os.path.join(write_dir, point_prefix + "_fin_data.csv"), index=False)
                fin_df_upsamp.to_csv(os.path.join(write_dir, point_prefix + "_fin_data_upsampled.csv"), index=False)
                raw_mesh.export(os.path.join(write_dir, point_prefix + "_rawfin_mesh.obj"))
                smoothed_mesh.export(os.path.join(write_dir, point_prefix + "_smoothed_fin_mesh.obj"))

                # add to wt vec
                wt_vec.append(smoothed_mesh.is_watertight)

    return wt_vec



if __name__ == '__main__':
    # root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    test = fin_mesh_wrapper(root)