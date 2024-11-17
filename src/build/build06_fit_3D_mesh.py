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
from sklearn import svm
# remeshing
import pyvista
import pyacvd
import pymeshfix
from sklearn.cluster import DBSCAN

def pyvista_to_mesh(mesh):
    v = mesh.points
    f = mesh.faces.reshape(-1, 4)[:, 1:4]
    return v, f


def mesh_to_pyvista(v, f):
    n, m = f.shape
    threes = np.full((n, 1), 3)
    face_arr = np.hstack((threes, f)).flatten()
    return pyvista.PolyData(v, face_arr)

def gs(X, row_vecs=True, norm=True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def mesh_cleanup(mesh_raw, target_verts=2500):

    v, f = mesh_raw.vertices, mesh_raw.faces
    # this can give a depreciation warning but it is fine
    mesh = mesh_to_pyvista(v, f)

    # target mesh resolution
    # target_verts = 2500

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(2)
    clus.cluster(target_verts)

    remesh = clus.create_mesh()

    v2, f2 = pyvista_to_mesh(remesh)

    # pymeshfix is often necessary here to get rid of non-manifold vertices
    v2, f2 = pymeshfix.clean_from_arrays(v2, f2)

    mesh_out = trimesh.Trimesh(vertices=v2, faces=f2)

    return mesh_out


def smooth_mesh_base(fin_mesh, fin_df, yolk_xyz, surf_center_o, fin_axes, base_axes, n_smooth_iters=25, ref_depth=-5):

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
    # sm_weight_vec = np.divide(np.exp(-(dist_metric - weight_hm) / weight_temperature), 1 + np.exp(-(dist_metric - weight_hm) / weight_temperature))
    dist_shifted = dist_metric - ref_depth
    dist_shifted[dist_shifted < 0] = 0
    norm_factor = ref_depth / np.log(0.1)
    sm_weight_vec = np.exp(-dist_shifted / norm_factor)
    # sm_weight_vec[sm_weight_vec < 0.05] = 0

    smoothed_mesh = fin_mesh.copy()

    for _ in tqdm(range(n_smooth_iters), "Smoothing mesh base..."):
        sm_temp = smoothed_mesh.copy()
        trimesh.smoothing.filter_laplacian(sm_temp, iterations=1)
        deltas = sm_temp.vertices - smoothed_mesh.vertices
        deltas_w = np.multiply(sm_weight_vec[:, np.newaxis], deltas)

        smoothed_mesh.vertices += deltas_w

    return smoothed_mesh

def flag_outliers_dbscan(fin_df):

    fin_points = fin_df[["XP", "YP", "ZP"]].to_numpy()
    # two passes, more stringent eps with single min, and then less stringent wti n>4
    eps0 = 10
    db0 = DBSCAN(eps=eps0, min_samples=4).fit(fin_points)
    labels0 = db0.labels_
    lb, lbc = np.unique(labels0, return_counts=True)
    lbc_top = lbc[0]
    lbc_frac = lbc / lbc_top
    outlier_labels = lb[lbc_frac < 0.1]
    outlier_flags0 = np.isin(labels0, outlier_labels)

    eps1 = 12
    db1 = DBSCAN(eps=eps1, min_samples=1).fit(fin_points)
    labels1 = db1.labels_
    lb, lbc = np.unique(labels1, return_counts=True)
    lbc_top = lbc[0]
    lbc_frac = lbc / lbc_top
    outlier_labels = lb[lbc_frac < 0.1]
    outlier_flags1 = np.isin(labels1, outlier_labels)

    return ~(outlier_flags0 | outlier_flags1)

# script that uses single class svd to ID and remove outliers
def flag_outlier_points(fin_df, outlier_frac=0.02):

    pts = fin_df[["YP", "ZP"]].to_numpy()
    # estimate gamma (scale coefficient) with variance heuristic
    # you may need to play with it to get good results, but this is a good starting point
    # Make gamma larger to get finer results, make it smaller to get smoother results (less holes)

    gamma = 1 / (3 * np.var(pts))

    # in this case, I'm making it larger becuase otherwise it fills up the hole
    gamma *= 2.5

    # print(f"gamma: {gamma:.5f}")

    # fit the model (can be quite slow!)
    model = svm.OneClassSVM(kernel="rbf", gamma=gamma, nu=outlier_frac)
    model.fit(pts)

    outlier_flags = model.predict(pts)

    # fin_df["inlier_flag"] = outlier_flags==1

    return outlier_flags==1

def get_yolk_distance(fin_object, yolk_dist_thresh=-5):

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

        # rerun fit to ensure we have an up-to-date estimate
        fin_object.fit_yolk_surface()
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

def get_base_axes(fin_object, fin_df, yolk_xyz, fin_axes, yolk_thresh=8):

    base_fin_points = fin_df.loc[np.abs(fin_df["yolk_dist"]) <= yolk_thresh, ["XP", "YP", "ZP"]].to_numpy()
    # base_fin_points_raw = fin_df.loc[np.abs(fin_df["yolk_dist"]) <= yolk_thresh, ["X", "Y", "Z"]].to_numpy()

    # calculate axis dims. Main one we care about is the AP axis ("YP")
    # axis_len_vec = np.max(base_fin_points, axis=0) - np.min(base_fin_points, axis=0)

    # find centroid
    ref10 = np.percentile(base_fin_points, 10, axis=0)
    ref90 = np.percentile(base_fin_points, 90, axis=0)
    point_center = (ref10 + ref90) / 2  # np.mean(base_fin_points_raw, axis=0)

    # use all proximal points to get a abetter estimate of AP center
    p_thresh = np.mean(fin_df["XP"])  # , 50)
    p_points = fin_df.loc[fin_df["XP"] <= p_thresh, ["YP"]].to_numpy()
    point_center[1] = np.mean(p_points)

    # rotate yolk surface to align with fin axes
    shift_ref_vec0 = np.mean(fin_df[["X", "Y", "Z"]].to_numpy(), axis=0)
    yolk_xyz_o = np.matmul(yolk_xyz - shift_ref_vec0, fin_axes.T)

    # find closest point on yolk surface
    surf_center_i = np.argmin(np.sqrt(np.sum((yolk_xyz_o - point_center) ** 2, axis=1)))
    surf_center = yolk_xyz[surf_center_i, :]  # this is the one we will use

    # define a local DV direction that is the cross product of the surface normal and the AP axis
    surf_normal_raw, _ = fin_object.calculate_tangent_plane(fin_object.yolk_surf_params, surf_center)
    if surf_normal_raw[2] > 0:
        surf_normal_raw = -surf_normal_raw

    # convert the normal vector to the biological axis space
    surf_normal = np.matmul(np.reshape(surf_normal_raw, (1, 3)), fin_axes.T)[0]
    surf_normal = surf_normal / np.linalg.norm(surf_normal)

    # calculate local DV
    dv_vec_base = np.cross(np.asarray([0, 1, 0]), surf_normal)
    dv_vec_base = dv_vec_base / np.linalg.norm(dv_vec_base)
    if dv_vec_base[0] < 0: # enforce alignment with PD
        dv_vec_base = -dv_vec_base

    # define a third vector that is orthogonal to AP and local (base) DV)
    # surf_vec_rel = np.cross(dv_vec_base,[0, 1, 0])
    # surf_vec_rel = surf_vec_rel / np.linalg.norm(surf_vec_rel)

    base_axes_raw = np.stack([dv_vec_base, np.asarray([0, 1, 0]), surf_normal], axis=0)
    base_axes = gs(base_axes_raw[::-1, :])[::-1, :]

    # shift centerpoint into the oriented frame of reference
    shift_ref_vec1 = np.mean(fin_df[["XP", "YP", "ZP"]].to_numpy(), axis=0)
    surf_center_o = np.matmul(surf_center - shift_ref_vec0, fin_axes.T)
    surf_center_b = np.matmul(surf_center_o - shift_ref_vec1, base_axes.T)

    return base_axes, surf_center_b, surf_center_o


def fin_mesh_wrapper(root, overwrite_flag=False, sampling_res=0.75, yolk_dist_thresh=-5, seg_type="tissue_only_best_model_tissue"):

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
    for file_ind, fp in enumerate(tqdm(fin_object_list[:6])):

        point_prefix = path_leaf(fp).replace("_fin_object.pkl", "")

        fin_object = FinData(data_root=root, name=point_prefix, tissue_seg_model=seg_type)

        test_path = os.path.join(write_dir, point_prefix + "_smoothed_fin_mesh.obj")
        if overwrite_flag or (not os.path.isfile(test_path)):
            print(f"processing {point_prefix}...")
            ############
            # calculate distances to yolk (and load fin_df)
            fin_df, yolk_xyz, fin_axes, nuclei_to_keep = get_yolk_distance(fin_object, yolk_dist_thresh=yolk_dist_thresh)

            if fin_df is not None:
                ############
                # find base centerpoint and orientation
                base_axes, surf_center_b, surf_center_o = get_base_axes(fin_object, fin_df, yolk_xyz, fin_axes)

                ################
                # flag outliers
                if fin_df.shape[0] > 10:
                    inlier_flags = flag_outliers_dbscan(fin_df)
                    inlier_nuclei = fin_df.loc[inlier_flags, "nucleus_id"].to_numpy()
                else:
                    inlier_nuclei = fin_df.loc[:, "nucleus_id"].to_numpy()

                ################
                # Upsample fin points
                fin_df_upsamp, nucleus_mask_array, nucleus_weight_array = upsample_fin_point_cloud(
                    fin_object, sample_res_um=sampling_res, root=root, points_per_nucleus=150)

                ################
                # Orient and resample fin points. FIT MESH
                # shift fin points
                fin_points_b = np.matmul(fin_df_upsamp.loc[:, ["XP", "YP", "ZP"]].to_numpy() - surf_center_o, base_axes.T)
                fin_df_upsamp[["XB", "YB", "ZB"]] = fin_points_b

                # get raw points
                nc_vec_us = fin_df_upsamp.loc[:, "nucleus_id"].to_numpy().astype(np.uint16)
                keep_filter0 = np.isin(nc_vec_us, nuclei_to_keep)
                keep_filter1 = np.isin(nc_vec_us, inlier_nuclei)
                keep_filter = keep_filter0 & keep_filter1
                fin_points = fin_df_upsamp.loc[keep_filter, ["XB", "YB", "ZB"]].to_numpy()

                # convert to point cloud format
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(fin_points)

                # resample points to be more spatially uniform
                sampled_points = pcd.voxel_down_sample(voxel_size=sampling_res)

                # fit fin mesh
                print("Fitting fin mesh...")
                mesh_alpha = 22
                watertight_flag = False
                fin_points_u = np.asarray(sampled_points.points)
                while not watertight_flag:
                    fin_mesh, raw_mesh, watertight_flag = fit_fin_mesh(fin_points_u, alpha=mesh_alpha)

                    mesh_alpha -= 1 # decrement
                    if mesh_alpha <= 18:
                        break

                ##############
                # Apply additional smoothing to mesh regions near the fin base
                smoothed_mesh = smooth_mesh_base(fin_mesh, fin_df, yolk_xyz, surf_center_o, fin_axes, base_axes)

                # use pyvista for final mesh cleanup step
                # mesh_out = mesh_cleanup(smoothed_mesh)

                # save
                fin_df.to_csv(os.path.join(write_dir, point_prefix + "_fin_data.csv"), index=False)
                fin_df_upsamp.to_csv(os.path.join(write_dir, point_prefix + "_fin_data_upsampled.csv"), index=False)
                raw_mesh.export(os.path.join(write_dir, point_prefix + "_raw_fin_mesh.obj"))
                smoothed_mesh.export(os.path.join(write_dir, point_prefix + "_smoothed_fin_mesh.obj"))

                np.save(os.path.join(write_dir, point_prefix + "_nucleus_id_array.npy"), nucleus_mask_array)
                np.save(os.path.join(write_dir, point_prefix + "_nucleus_weight_array.npy"), nucleus_weight_array)
                # add to wt vec

                wt_vec.append(smoothed_mesh.is_watertight)

    return wt_vec



if __name__ == '__main__':
    # root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

    test = fin_mesh_wrapper(root)