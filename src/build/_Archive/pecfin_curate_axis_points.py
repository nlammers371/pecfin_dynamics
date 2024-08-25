# from aicsimageio import AICSImage
import numpy as np
import napari
import os
from src.utilities.functions import path_leaf
from glob2 import glob
import skimage.io as skio
import pandas as pd
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.neural_network import MLPClassifier, MLPRegressor
import time
import vispy.color
import zarr
from scipy.spatial import Delaunay
from skimage.measure import regionprops
from tqdm import tqdm
from scipy.spatial import distance_matrix
from src.utilities.surface_axes_functions import *
import scipy


def fit_yolk_surface():

    yolk_data = labels_df.loc[(labels_df['fin_label_curr'] == 2) & (labels_df['fin_proximity_flag'] == 1), ["X", "Y", "Z"]].to_numpy()

    # best-fit quadratic curve. Currently of form z = f(x,y). This will work so long as yolk is not vertical in z plane
    test_y = np.arange(np.percentile(yolk_data[:,1], 0.5), np.percentile(yolk_data[:,1], 99.5), 1)
    test_x = np.arange(np.percentile(yolk_data[:,0], 0.5), np.percentile(yolk_data[:,0], 99.5), 1)
    XX, YY = np.meshgrid(test_x, test_y)
    # X = XX.ravel()
    # Y = YY.ravel()

    def loss_fun(C, data=yolk_data):
        loss = quadratic_z_loss(C, data)
        return loss

    c0 = np.asarray([-3, 0.01, 0.015, 0.001, 0.005, 0.01])
    C_yolk = scipy.optimize.least_squares(loss_fun,
                                          c0, bounds=([-np.inf, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
    C_yolk = C_yolk.x
    A_in = make_quadratic_array(yolk_data[:, :2])
    # C_yolk, _, _, _ = scipy.linalg.lstsq(A_in, yolk_data[:, 2])
    # # evaluate it on a grid
    # A_out = make_quadratic_array(np.c_[X, Y])
    Z = predict_z(A_in, C_yolk)
    # ZZ = np.reshape(Z, XX.shape)
    # surf_array = np.stack((ZZ, YY, XX), axis=-1)

    vertices = np.c_[Z, yolk_data[:, 1], yolk_data[:, 0]]
    # Generate the Delaunay triangulation to create faces
    tri = Delaunay(yolk_data[:, :2])
    faces = tri.simplices

    return vertices, faces

def get_cbrewer_paired0():
    paired_colors = np.asarray([
        [100, 100, 100],
        [31, 120, 180],  # blue
        [51, 160, 44],  # green
        [227, 26, 28],  # red
        [166, 206, 227],  # light blue
        [178, 223, 138],  # light green
        [251, 154, 153],  # light red
        # [253, 191, 111],  # light orange
        # [255, 127, 0],  # orange
        # [202, 178, 214],  # light purple
        # [106, 61, 154],  # purple
        # [55, 255, 153],  # light yellow
        # [177, 89, 40]  # brown
    ]) / 255

    paired_colormap = vispy.color.Colormap(paired_colors)
    return paired_colormap

def get_cbrewer_paired1():
    paired_colors = np.asarray([
        [100, 100, 100],
        [106, 61, 154],  # purple
        [51, 160, 44],  # green
        [227, 26, 28],  # red
        [202, 178, 214],  # light purple
        [178, 223, 138],  # light green
        [251, 154, 153],  # light red
        # [253, 191, 111],  # light orange
        # [255, 127, 0],  # orange

        # [55, 255, 153],  # light yellow
        # [177, 89, 40]  # brown
    ]) / 255
    paired_colormap = vispy.color.Colormap(paired_colors)
    return paired_colormap
def update_point_axes(axis_df, point_df, tissue_flag=1):

    tissue_indices = np.where(labels_df["fin_label_curr"] == tissue_flag)[0]

    # generate AP vals
    ap_vec = (axis_df.loc[1, ["X", "Y", "Z"]].to_numpy() - axis_df.loc[4, ["X", "Y", "Z"]].to_numpy()).astype(np.float64)
    ap_vec = ap_vec / np.linalg.norm(ap_vec)
    ap_vals = np.matmul(point_df.loc[tissue_indices, ["X", "Y", "Z"]].to_numpy(), ap_vec[np.newaxis, :].T).astype(np.float64)

    # DV
    dv_vec = (axis_df.loc[3, ["X", "Y", "Z"]].to_numpy() - axis_df.loc[6, ["X", "Y", "Z"]].to_numpy()).astype(np.float64)
    dv_vec = dv_vec / np.linalg.norm(dv_vec)
    dv_vals = np.matmul(point_df.loc[tissue_indices, ["X", "Y", "Z"]].to_numpy(), dv_vec[np.newaxis, :].T).astype(np.float64)

    # LR
    lr_vec = (axis_df.loc[2, ["X", "Y", "Z"]].to_numpy() - axis_df.loc[5, ["X", "Y", "Z"]].to_numpy()).astype(np.float64)
    lr_vec = lr_vec / np.linalg.norm(lr_vec)
    lr_vals = np.matmul(point_df.loc[tissue_indices, ["X", "Y", "Z"]].to_numpy(), lr_vec[np.newaxis, :].T).astype(np.float64)

    # normalize
    axis_array = np.c_[ap_vals, dv_vals, lr_vals]
    axis_array = axis_array / np.std(axis_array[:, 0], axis=0)

    # store
    point_df.loc[tissue_indices, ["AP", "DV", "LR"]] = axis_array

    return point_df

def get_axis_predictions(axis_df, mdl, mlp_df=None, tissue_flag=1, n_train=500):

    # filter for correct tissue
    tissue_indices = np.where(point_df["label_pd"] == tissue_flag)[0]
    both_flag = True
    if mlp_df is None:
        both_flag = False
        # extract training data
        train_indices = np.random.choice(tissue_indices, n_train, replace=True)
        mlp_df = point_df.loc[train_indices, :]

    feature_cols = []
    feature_cols += [c for c in point_df.columns if "feat" in c]  # + ["well_num", "time_int", "date_norm"]
    Y_train = mlp_df.loc[:, ["AP", "DV", "LR"]].to_numpy()
    X_train = mlp_df.loc[:, feature_cols].to_numpy()

    # train
    print("Updating axis predictions...")
    start = time.time()
    mdl = mdl.fit(X_train, Y_train)
    print(time.time() - start)

    # get new predictions
    X_pd = point_df.loc[tissue_indices, feature_cols].to_numpy()
    Y_pd_nonlin = mdl.predict(X_pd)

    # predict AP axis first
    xyz_array = point_df.loc[tissue_indices, ["X", "Y", "Z"]].to_numpy()
    ap_vec = np.linalg.lstsq(xyz_array, Y_pd_nonlin[:, 0], rcond=-1)[0]
    ap_vec = ap_vec / np.linalg.norm(ap_vec)
    dv_vec = np.linalg.lstsq(xyz_array, Y_pd_nonlin[:, 1], rcond=-1)[0]
    dv_vec = dv_vec / np.linalg.norm(dv_vec)
    lr_vec = np.linalg.lstsq(xyz_array, Y_pd_nonlin[:, 2], rcond=-1)[0]
    lr_vec = lr_vec / np.linalg.norm(lr_vec)

    # find closest orthogonal set
    axes_raw = np.c_[ap_vec, dv_vec, lr_vec]
    axes_orth = gs(axes_raw)

    # update df
    centroid = axis_df.loc[0, ["X", "Y", "Z"]].to_numpy().astype(float)
    axis_df.loc[0, ["X_pd", "Y_pd", "Z_pd"]] = centroid
    # get AP predictions
    axis_df.loc[1, ["X_pd", "Y_pd", "Z_pd"]] = centroid + 0.66*handle_scale_global * axes_orth[:, 0]
    axis_df.loc[4, ["X_pd", "Y_pd", "Z_pd"]] = centroid - 0.66*handle_scale_global * axes_orth[:, 0]

    # get DV predictions
    axis_df.loc[3, ["X_pd", "Y_pd", "Z_pd"]] = centroid + 0.66*handle_scale_global * axes_orth[:, 1]
    axis_df.loc[6, ["X_pd", "Y_pd", "Z_pd"]] = centroid - 0.66*handle_scale_global * axes_orth[:, 1]

    # get LR predictions
    axis_df.loc[2, ["X_pd", "Y_pd", "Z_pd"]] = centroid + 0.66*handle_scale_global * axes_orth[:, 2]
    axis_df.loc[5, ["X_pd", "Y_pd", "Z_pd"]] = centroid - 0.66*handle_scale_global * axes_orth[:, 2]

    if both_flag:
        # get AP predictions
        axis_df.loc[1, ["X", "Y", "Z"]] = centroid + handle_scale_global * axes_orth[:, 0]
        axis_df.loc[4, ["X", "Y", "Z"]] = centroid - handle_scale_global * axes_orth[:, 0]

        # get DV predictions
        axis_df.loc[3, ["X", "Y", "Z"]] = centroid + handle_scale_global * axes_orth[:, 1]
        axis_df.loc[6, ["X", "Y", "Z"]] = centroid - handle_scale_global * axes_orth[:, 1]

        # get LR predictions
        axis_df.loc[2, ["X", "Y", "Z"]] = centroid + handle_scale_global * axes_orth[:, 2]
        axis_df.loc[5, ["X", "Y", "Z"]] = centroid - handle_scale_global * axes_orth[:, 2]


    return axis_df


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

def update_axes(axis_df, selected_index, layer):

    data = layer.data
    new_position = layer.data[selected_index].copy()

    new_vec = new_position - data[0, :]
    new_vec = new_vec / np.sqrt(np.sum((new_vec) ** 2))
    if selected_index > 0:
        # update orientation of target axis
        if selected_index < 4:
            alt_index = selected_index + 3
        else:
            alt_index = selected_index - 3

        scale_factor = 1
        if layer.name == "fin axis points":
            scale_factor = 0.6
        new_pos0 = data[0, :] + scale_factor * new_vec * handle_scale_global
        new_pos1 = data[0, :] - scale_factor * new_vec * handle_scale_global
        data[selected_index, :] = new_pos0
        data[alt_index, :] = new_pos1

        D = -np.dot(new_vec, data[0, :])

        # we also need to adjust the other axes for consistency--they all need to be orthogonal
        options = [i for i in range(1, 4) if i not in [alt_index, selected_index]]
        ind = np.random.choice(options, 1)[0]
        curr_point = data[ind, :]
        # v = curr_point - data[0, :]
        dist = np.dot(curr_point, new_vec) + D

        raw_point = curr_point - new_vec * dist
        vec = raw_point - data[0, :]
        vec = vec / np.sqrt(np.sum((vec) ** 2))
        data[ind, :] = data[0, :] + scale_factor * vec * handle_scale_global
        data[ind + 3, :] = data[0, :] - scale_factor * vec * handle_scale_global

        last_i = [i for i in range(1, 4) if i not in [alt_index, selected_index, ind]][0]
        vec1 = np.cross(vec, new_vec)
        vec1 = vec1 / np.sqrt(np.sum((vec1) ** 2))
        data[last_i, :] = data[0, :] + scale_factor * vec1 * handle_scale_global
        data[last_i + 3, :] = data[0, :] - scale_factor * vec1 * handle_scale_global

        layer.data = data

        if layer.name == "body axis points":
            axis_df.loc[axis_df["tissue_name"] == "body", ["X", "Y", "Z"]] = data

        elif layer.name == "fin axis points":
            axis_df.loc[axis_df["tissue_name"] == "fin", ["X", "Y", "Z"]] = data

        return axis_df

def sample_reference_points(mlp_df, axis_df, point_df, npoints=50):

    labels_u, labels_to = np.unique(axis_df.loc[:, "axis_label_pd"], return_inverse=True)
    labels_per = np.ceil(npoints / len(labels_u)).astype(int)
    ref_indices = []
    for l, lb in enumerate(labels_u):
        lbi = np.where(labels_u == lb)[0]
        lb_indices = np.where(labels_to == lbi)[0]
        ns = np.min([labels_per, len(lb_indices)])
        if ns > 0:
            # _, temp_indices = farthest_point_sample(axis_df.loc[lb_indices, ["Z", "Y", "X"]].to_numpy(), ns)
            temp_indices = np.random.choice(lb_indices, size=ns, replace=False)
            ref_indices += temp_indices.tolist()

    # transfer labels
    axis_df.loc[ref_indices, "axis_label_curr"] = axis_df.loc[ref_indices, "axis_label_pd"]

    # add to training set
    mlp_df_temp = point_df.loc[axis_df["axis_label_curr"] != 0]
    mlp_df_temp.loc[:, "axis_label_curr"] = axis_df.loc[axis_df["axis_label_curr"] != 0, "axis_label_curr"].copy()

    if len(mlp_df) > 0:
        mlp_df = pd.concat([mlp_df, mlp_df_temp])
        mlp_df = mlp_df.drop_duplicates(keep="first", subset=["experiment_date", "well_num", "time_int", "nucleus_id"], ignore_index=True)
    else:
        mlp_df = mlp_df_temp.copy()

    return mlp_df, axis_df

def strip_dummy_cols(df):
    cols = df.columns
    keep_cols = [col for col in cols if "Unnamed" not in col]
    df = df[keep_cols]
    return df

def fit_mlp(axis_df, mdl, mlp_df):

    feature_cols = []
    feature_cols += [c for c in mlp_df.columns if "feat" in c] + ["label_pd"]  # + ["well_num", "time_int", "date_norm"]
    X_train = mlp_df.loc[:, feature_cols]

    Y_train = mlp_df.loc[:, "axis_label_curr"].to_numpy()

    print("Updating axis predictions...")
    mdl = mdl.fit(X_train, Y_train)

    # get new predictions
    X_pd = point_df.loc[:, feature_cols]

    Y_pd = mdl.predict(X_pd)

    if axis_df is not None:
        axis_df.loc[:, "axis_label_pd"] = Y_pd
    else:
        pass

    return axis_df, Y_pd, mdl

def update_mlp_data(axis_df, mlp_df, point_df, intra_well_only):

    # generate wide feature DF for classifier training
    mlp_df_temp = point_df.loc[axis_df["axis_label_curr"] != 0]
    mlp_df_temp.loc[:, "axis_label_curr"] = axis_df.loc[axis_df["axis_label_curr"] != 0, "axis_label_curr"].copy()
    mlp_df_temp["label_pd"] = axis_df["label_pd"]

    well_num = point_df.loc[0, "well_num"]
    # remove observations from different wells if desired
    if intra_well_only and (len(mlp_df) > 0):
        mlp_df = mlp_df.loc[mlp_df["well_num"] == well_num]

    if len(mlp_df) > 0:
        mlp_df = pd.concat([mlp_df_temp, mlp_df])
        mlp_df = mlp_df.drop_duplicates(keep="first", subset=["experiment_date", "well_num", "time_int", "nucleus_id"],
                                        ignore_index=True)
    else:
        mlp_df = mlp_df_temp.copy()

    mlp_df = strip_dummy_cols(mlp_df)
    mlp_df = mlp_df.dropna()

    return mlp_df

def load_mlp_data(root, mlp_arch, n_points_per_set=400, intra_well_only=False):

    point_path = os.path.join(root, "point_cloud_data", "point_features", seg_type_global, "")
    if intra_well_only:
        n_points_per_set = 400

    axis_data_path = os.path.join(root, "point_cloud_data", "axis_inference", seg_type_global, "")

    # get list of extant labeled datasets
    df_list = glob(axis_data_path + "*_axes.csv")

    # load each in and sample
    df_out = []
    if len(df_list) > 0:
        for df_path in df_list:
            # load labels
            axis_df = pd.read_csv(df_path)

            # load points
            point_name = path_leaf(df_path).replace("axes", "points_features")
            point_df_temp = pd.read_csv(os.path.join(point_path, point_name))

            if "approved_flag" in axis_df.columns:
                if axis_df.loc[0, "approved_flag"]:
                    point_df_temp = update_point_axes(axis_df, point_df_temp, tissue_flag=3)
                    options = np.where(point_df_temp["label_pd"] == 3)[0]
                    sample_indices = np.random.choice(options, size=n_points_per_set, replace=True)
                    df_temp = point_df_temp.loc[sample_indices]
                    df_out.append(df_temp)

        if len(df_out) > 0:
            df_out = pd.concat(df_out, axis=0, ignore_index=True)

    # initialize model
    mdl = MLPRegressor(max_iter=5000, hidden_layer_sizes=mlp_arch)
    return df_out, mdl

def get_fin_layers(labels_df, prob_zarr, mask_zarr, pd_mask, dist_thresh=50):

    xyz_array_fin = labels_df.loc[labels_df["fin_label_curr"] == 1, ["X", "Y", "Z"]].to_numpy()
    xyz_array = labels_df.loc[:, ["X", "Y", "Z"]].to_numpy()

    dist_mat = distance_matrix(xyz_array, xyz_array_fin)
    fin_dist = np.min(dist_mat, axis=1)  # just using simple euclidean distance for now

    # filter for nearby surface nuclei
    dist_ft = fin_dist <= dist_thresh
    labels_df.loc[:, "fin_proximity_flag"] = dist_ft
    close_ids = labels_df.loc[dist_ft, "nucleus_id"].to_numpy()

    # get subset mask
    label_mask_fin = np.zeros_like(mask_zarr)
    id_mask = np.isin(mask_zarr, close_ids)
    label_mask_fin[id_mask] = pd_mask[id_mask]

    # get filtered zarr
    prob_zarr_fin = np.zeros_like(prob_zarr)
    prob_zarr_fin = prob_zarr_fin + np.min(prob_zarr)
    for t in range(prob_zarr_fin.shape[0]):
        prob_zarr_fin[t][id_mask] = prob_zarr[t][id_mask]

    # make region-specific mask
    mask_zarr_fin = np.zeros_like(mask_zarr)
    mask_zarr_fin[id_mask] = mask_zarr[id_mask]

    return label_mask_fin, prob_zarr_fin, mask_zarr_fin

def load_zarr_data(root, seg_model, experiment_date, file_prefix, time_int):

    # path to raw data
    raw_zarr_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, file_prefix + ".zarr")
    mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, experiment_date, file_prefix + "_mask_aff.zarr")
    prob_zarr_path = os.path.join(root, "built_data", "cellpose_output", seg_model, experiment_date,
                                  file_prefix + "_probs.zarr")

    data_zarr = zarr.open(raw_zarr_path, mode="r")
    mask_zarr = zarr.open(mask_zarr_path, mode="r")
    prob_zarr = zarr.open(prob_zarr_path, mode="r")

    # convert scale vec to tuple
    scale_vec = data_zarr.attrs["voxel_size_um"]
    scale_vec = tuple(scale_vec)

    # load the specific time point we want
    time_range = np.arange(np.max([0, time_int-3]), np.min([len(prob_zarr), time_int+4]))
    im_prob = prob_zarr[time_range]
    im_mask = mask_zarr[time_int]

    return im_prob, im_mask, scale_vec


def load_points_and_labels(root, file_prefix, time_int):

    point_prefix = file_prefix + f"_time{time_int:04}"
    # check to see if fin class object exists
    fin_path = os.path.join(root, "point_cloud_data", "fin_objects", "")

    # check for point cloud dataset
    point_path = os.path.join(root, "point_cloud_data", "point_features", seg_type_global, "")
    axis_path_out = os.path.join(root, "point_cloud_data", "axis_inference", seg_type_global, "")

    if not os.path.isdir(axis_path_out):
        os.makedirs(axis_path_out)

    point_df_temp = pd.read_csv(point_path + point_prefix + "_points_features.csv")
    point_df_temp = strip_dummy_cols(point_df_temp)
    point_df = point_df_temp.copy()

    # check to see if manual tissue labels exist (otherwise we'll use model predictions)
    curation_path = os.path.join(root, "point_cloud_data", "fin_segmentation", "")
    dir_list = sorted(glob(curation_path + "*"))
    label_path = ""
    labels_df = []
    for dir_path in dir_list:
        if os.path.isfile(dir_path + point_prefix + "_labels.csv"):
            labels_df = pd.read_csv(curation_path + point_prefix + "_labels.csv")
            labels_df = strip_dummy_cols(labels_df)
            label_path = dir_path + point_prefix + "_labels.csv"

    if len(labels_df) == 0:
        keep_cols = [col for col in point_df.columns if "feat" not in col]
        labels_df = point_df.loc[:, keep_cols]
        labels_df["fin_label_curr"] = point_df["label_pd"] + 1
        labels_df["approved_flag"] = False

    # check for pre-existing labels DF
    if os.path.isfile(axis_path_out + point_prefix + "_axes.csv"):
        axis_df = pd.read_csv(axis_path_out + point_prefix + "_axes.csv")
        axis_df = strip_dummy_cols(axis_df)

    else:
        # initialize dataframe
        xyz_body = labels_df.loc[labels_df["fin_label_curr"] == 3, ["X", "Y", "Z"]].to_numpy().astype(float)
        xyz_fin = labels_df.loc[labels_df["fin_label_curr"] == 1, ["X", "Y", "Z"]].to_numpy().astype(float)

        id_vec = ["C", "A", "L", "D", "P", "R", "V"] + ["C", "Pr", "L", "D", "Di", "R", "V"]
        tissue_id_vec = [3]*7 + [1]*7
        tissue_name_vec = ["body"]*7 + ["fin"]*7
        point_id_float = np.asarray([0, 1, 2, 3, 4, 5, 6] + [0, 1, 2, 3, 4, 5, 6]) / 6

        axis_df = pd.DataFrame(id_vec, columns=["point_id"])

        axis_df["point_id_float"] = point_id_float
        axis_df["tissue_id"] = tissue_id_vec
        axis_df["tissue_name"] = tissue_name_vec
        axis_df[["X", "Y", "Z"]] = np.nan #np.concatenate((body_centroid[np.newaxis, :], xyz_min_array, xyz_max_array), axis=0)
        axis_df[["X_pd", "Y_pd", "Z_pd"]] = np.nan

        axis_df.loc[0, ["X", "Y", "Z"]] = np.mean(xyz_body, axis=0)
        axis_df.loc[7, ["X", "Y", "Z"]] = np.mean(xyz_fin, axis=0)
        axis_df[["approved_flag"]] = False

    return point_df, labels_df, axis_df, point_prefix, axis_path_out

def get_pca_axis_prediction(axis_df, tissue_flag=1):
    
    xyz_array = labels_df.loc[labels_df["fin_label_curr"] == tissue_flag, ["X", "Y", "Z"]].to_numpy().astype(float)
    
    point_pca = PCA(n_components=3)
    point_pca.fit(xyz_array)

    # pca_array_body = point_pca.transform(xyz_body)
    axes = point_pca.components_
    centroid = np.mean(xyz_array, axis=0)

    scale_factor = 0.6
    if tissue_flag > 1:
        scale_factor = 1

    xyz_min_array = -handle_scale_global * scale_factor * axes + centroid
    xyz_max_array = handle_scale_global * scale_factor * axes + centroid

    xyz_min_array_pd = -0.66*handle_scale_global * scale_factor * axes + centroid
    xyz_max_array_pd = 0.66*handle_scale_global * scale_factor * axes + centroid
    
    axis_df.loc[axis_df["tissue_id"] == tissue_flag, ["X", "Y", "Z"]] = np.concatenate((centroid[np.newaxis, :], xyz_min_array, xyz_max_array), axis=0)
    axis_df.loc[axis_df["tissue_id"] == tissue_flag, ["X_pd", "Y_pd", "Z_pd"]] = np.concatenate((centroid[np.newaxis, :], xyz_min_array_pd, xyz_max_array_pd), axis=0)

    d_filter = (axis_df["point_id"] == "D") & (axis_df["tissue_id"] == tissue_flag).to_numpy()
    v_filter = (axis_df["point_id"] == "V") & (axis_df["tissue_id"] == tissue_flag).to_numpy()
    if axis_df.loc[d_filter, "Z"].to_numpy()[0] > axis_df.loc[v_filter, "Z"].to_numpy()[0]:
        top = axis_df.iloc[v_filter, 1:].copy()
        bottom = axis_df.iloc[d_filter, 1:].copy()
        axis_df.iloc[d_filter, 1:] = bottom
        axis_df.iloc[v_filter, 1:] = top

    return axis_df

def toggle_body_axis_approval(viewer):
    global axis_df

    # switch approval of body axis points
    axis_df.loc[axis_df["tissue_name"] == "body", "approved_flag"] = ~axis_df.loc[axis_df["tissue_name"] == "body", "approved_flag"]

    if axis_df.loc[1, "approved_flag"]:
        print("Body axes approved...")
        body_axis_layer.border_width = 0.1
    else:
        print("Body axes disapproved...")
        body_axis_layer.border_width = 0

def toggle_fin_axis_approval(viewer):
    global axis_df

    # switch approval of body axis points
    axis_df.loc[axis_df["tissue_name"] == "fin", "approved_flag"] = ~axis_df.loc[
        axis_df["tissue_name"] == "fin", "approved_flag"]

    if axis_df.loc[8, "approved_flag"]:
        print("Fin axes approved...")
        fin_axis_layer.border_width = 0.1
    else:
        print("Fin axes disapproved...")
        fin_axis_layer.border_width = 0

def toggle_label_approval(viewer):
    global axis_df

    # switch approval of body axis points
    labels_df.loc[:, "approved_flag"] = ~labels_df.loc[:, "approved_flag"]

    if labels_df.loc[0, "approved_flag"]:
        print("Tissue labels approved...")
        label_layer_fin.opacity = 0.45
    else:
        print("Tissue labels disapproved...")
        label_layer_fin.opacity = 0.25

def on_mouse_drag(layer, event):
    global original_position, axis_df, point_df
    # drag_started = True
    original_position = None
    drag_started = False
    # On press, record the initial position of the selected point
    if event.type == 'mouse_press' and event.button == 1 and layer.mode == "select":
        selected_index = layer.get_value(event.position, world=True, view_direction=event.view_direction,
                                         dims_displayed=event.dims_displayed)
        original_position = layer.data[selected_index].copy()

        # Yield to the next callback in the sequence (this allows the drag operation to continue)
        yield

        # # # After dragging, check if the point was moved
        while event.type == 'mouse_move':
            drag_started = True
            yield

        if event.type == 'mouse_release':
            new_position = layer.data[selected_index].copy()
            data = layer.data
            if drag_started and not np.array_equal(original_position, new_position):
                pass
                # if layer.name == "body axis points":
                #     axis_df = update_axes(axis_df, selected_index, layer)
                #     point_df = update_point_axes(axis_df=axis_df, point_df=point_df, tissue_flag=2)
                #     axis_df = get_axis_predictions(axis_df=axis_df, mdl=mdl, tissue_flag=2)
                #     body_axis_layer_pd.data = axis_df[["X_pd", "Y_pd", "Z_pd"]].to_numpy()
            else:
                if selected_index > 0:
                    swap0 = data[selected_index, :].copy()
                    if selected_index < 4:
                        alt_index = selected_index + 3
                    else:
                        alt_index = selected_index - 3
                    swap1 = data[alt_index, :].copy()

                    data[selected_index, :] = swap1
                    data[alt_index, :] = swap0
                    layer.data = data

                    # update axis data frames and update prediction
            # if layer.name == "body axis points":
            axis_df = update_axes(axis_df, selected_index, layer)
            # point_df = update_point_axes(axis_df=axis_df, point_df=point_df, tissue_flag=2)

def label_update_function(event):
    global axis_df #mlp_df, mdl

    if event.type == 'paint':

        # get event coordinates and values
        zv = event.value[0][0][0]
        yv = event.value[0][0][1]
        xv = event.value[0][0][2]
        for i in range(1, len(event.value)):
            zv = np.concatenate((zv, event.value[i][0][0]))
            yv = np.concatenate((yv, event.value[i][0][1]))
            xv = np.concatenate((xv, event.value[i][0][2]))
        mask_coords = tuple([zv, yv, xv])
        nc_ids = mask_zarr_fin[mask_coords]

        ft = nc_ids != 0

        mask_coords = list(mask_coords)
        mask_coords[0] = mask_coords[0][ft]
        mask_coords[1] = mask_coords[1][ft]
        mask_coords[2] = mask_coords[2][ft]
        mask_coords = tuple(mask_coords)

        if len(mask_coords) > 0:
            nucleus_id = np.unique(mask_zarr_fin[mask_coords])
            new_value = label_layer_fin.data[mask_coords][0]

            # update layer
            label_layer_fin.data[np.isin(mask_zarr_fin, nucleus_id)] = new_value
            labels_df.loc[np.isin(labels_df["nucleus_id"], nucleus_id), "fin_label_curr"] = new_value

            # updated training DF
            # mlp_df = point_df.loc[labels_df["fin_label_curr"] != 0]
            # mlp_df.reset_index(inplace=True, drop=True)
            # mlp_df.loc[:, "fin_label_curr"] = labels_df.loc[labels_df["fin_label_curr"] != 0, "fin_label_curr"].to_numpy()

        else:
            pass

        # update label layer
        lb_data = label_layer_fin.data
        lb_data[mask_zarr_fin == 0] = 0
        label_layer_fin.data = lb_data

        # update fin axes
        axis_df = get_pca_axis_prediction(axis_df=axis_df, tissue_flag=1)
        toggle_fin_axis_approval(viewer)
        fin_axis_layer.data = axis_df.loc[axis_df["tissue_name"] == "fin", ["Z", "Y", "X"]].to_numpy()

# def fit_tissue_mlp(viewer):
#     # update training DF
#     if (mlp_df.shape[0] > 10) & (train_counter > 2):
#
#         _, Y_pd, _ = fit_mlp(None, mdl, mlp_df)
#
#         labels_df["fin_label_pd"] = Y_pd
#         labels_u = [1, 2, 3, 4]
#         pd_mask = pd_layer.data
#         for lb in labels_u:
#             pd_ids = labels_df.loc[labels_df["fin_label_pd"] == lb, "nucleus_id"].values
#             pd_mask[np.isin(mask_zarr, pd_ids)] = lb
#
#         pd_layer.data = pd_mask
def curate_pec_fins(root, experiment_date, well_num, seg_model, seg_type, time_int=0, mlp_arch=None,
                         handle_scale=150, intra_well_only=False):

    if mlp_arch is None:
        mlp_arch = (16,)

    # initialize global variables
    global mlp_df, mdl, point_df, axis_df, Y_probs, mask_zarr_fin, label_mask, mask_zarr_fin, label_layer_fin, viewer,\
        seg_type_global, handle_scale_global, body_axis_layer, fin_axis_layer, labels_df

    seg_type_global = seg_type
    handle_scale_global = handle_scale

    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"

    prob_zarr, mask_zarr, scale_vec = load_zarr_data(root, seg_model, experiment_date, file_prefix, time_int)

    # load point features and labels
    point_df, labels_df, axis_df, point_prefix, axis_path_out = load_points_and_labels(root, file_prefix, time_int)
    # point_df["label_pd"] = point_df.loc[:, "label_pd"]

    # load model
    # mlp_df_all, mdl = load_mlp_data(root=root, mlp_arch=mlp_arch)  # use info from othar datasets to make an initial prediction

    # perform initial fit if we have enough local or cross-well training data
    if False: #(len(mlp_df_all) > 10) and not intra_well_only:
        pass #axis_df = get_axis_predictions(axis_df=axis_df, mdl=mdl, mlp_df=mlp_df_all, tissue_flag=2)
    else:
        if not axis_df.loc[0, "approved_flag"]:
            axis_df = get_pca_axis_prediction(axis_df=axis_df, tissue_flag=3)
        if not axis_df.loc[7, "approved_flag"]:
            axis_df = get_pca_axis_prediction(axis_df=axis_df, tissue_flag=1)

    # add AP coordinates for body
    # points_df = update_point_axes(axis_df, point_df=point_df, tissue_flag=2)

    # initialize viewer
    viewer = napari.Viewer()
    viewer.add_image(prob_zarr, colormap="gray", name="probabilities",
                     scale=scale_vec, contrast_limits=(-4, np.percentile(prob_zarr, 99.8)))

    # show tissue labels for reference
    pd_mask = np.zeros_like(mask_zarr)
    for lb in [1, 2, 3, 4]:
        pd_ids = labels_df.loc[labels_df["fin_label_curr"] == lb, "nucleus_id"].values
        pd_mask[np.isin(mask_zarr, pd_ids)] = lb

    # get fin-specific layers
    label_mask_fin, prob_zarr_fin, mask_zarr_fin = get_fin_layers(labels_df, prob_zarr, mask_zarr, pd_mask)

    vertices, faces = fit_yolk_surface()

    # Add the surface layer to the viewer
    viewer.add_surface((vertices, faces, vertices[:, 1]))

    # make fin-focused probability later
    prob_layer_fin = viewer.add_image(prob_zarr_fin, name="probabilities (fin region)", colormap="gray", scale=scale_vec, visible=False,
                                      contrast_limits=(-4, np.percentile(prob_zarr, 99.8)))
    # show tissue predictions
    label_opacity = 0.25
    if np.all(labels_df.loc[:, "approved_flag"]):
        label_opacity = 0.5
    label_layer = viewer.add_labels(pd_mask, scale=scale_vec, name='tissue predictions (static)', opacity=label_opacity, visible=False)
    label_layer_fin = viewer.add_labels(label_mask_fin, scale=scale_vec, name='tissue predictions (fin region)', opacity=label_opacity,
                                    visible=False)

    paired_colors0 = get_cbrewer_paired0()
    paired_colors1 = get_cbrewer_paired1()

    # body_axis_layer_pd = viewer.add_points(axis_df.loc[:, ["Z_pd", "Y_pd", "X_pd"]].to_numpy(),
    #                                        name='body axis points (predicted)',
    #                                        size=10, features=axis_df.loc[:, ["point_id", "point_id_float"]],
    #                                        face_color="point_id_float", face_colormap=paired_colors, visible=True,
    #                                        out_of_slice_display=True, symbol="diamond")
    body_border = 0
    fin_border = 0
    if axis_df.loc[1, "approved_flag"] == True:
        body_border = 0.1

    if axis_df.loc[7, "approved_flag"] == True:
        fin_border = 0.05

    body_axis_layer = viewer.add_points(axis_df.loc[axis_df["tissue_name"] == "body", ["Z", "Y", "X"]].to_numpy(), name='body axis points',
                                        size=25, features=axis_df.loc[axis_df["tissue_name"] == "body", ["point_id", "point_id_float"]],
                                        border_color="black", face_color="point_id_float", face_colormap=paired_colors0,
                                        visible=True, out_of_slice_display=True, border_width=body_border)

    fin_axis_layer = viewer.add_points(axis_df.loc[axis_df["tissue_name"] == "fin", ["Z", "Y", "X"]].to_numpy(),
                                        name='fin axis points',
                                        size=15, features=axis_df.loc[axis_df["tissue_name"] == "fin", ["point_id", "point_id_float"]],
                                        border_color="black", face_color="point_id_float", face_colormap=paired_colors1,
                                        visible=True, out_of_slice_display=True, border_width=fin_border, symbol="diamond")

    # connect to event trigger function
    body_axis_layer.mouse_drag_callbacks.append(on_mouse_drag)
    fin_axis_layer.mouse_drag_callbacks.append(on_mouse_drag)
    label_layer_fin.events.paint.connect(label_update_function)

    # add key bindings
    viewer.bind_key("b", toggle_body_axis_approval)
    viewer.bind_key("f", toggle_fin_axis_approval)
    viewer.bind_key("t", toggle_label_approval)

    napari.run()

    approved_data_path = os.path.join(root, "point_cloud_data", "approved_fin_data", "")
    # save axes
    if not np.all(axis_df["approved_flag"]):
        wait = input("Axes are not approved yet. Press a to approve labels for training. \nOtherwise, press Enter then Enter.")
        if 'a' in wait:
            print("Axes approved...")
            axis_df[["approved_flag"]] = True
    # save
    print("Saving axis data...")
    axis_df.to_csv(axis_path_out + point_prefix + "_axes.csv", index=False)
    if np.all(axis_df["approved_flag"]):
        axis_df["tissue_model"] = seg_type_global
        axis_df.to_csv(approved_data_path + point_prefix + "_axes.csv", index=False)

    # save labels
    if not np.all(labels_df["approved_flag"]):
        wait = input(
            "Tissue labels are not approved yet. Press a to approve labels for training. \nOtherwise, press Enter then Enter.")
        if 'a' in wait:
            labels_df[["approved_flag"]] = True
        else:
            print("Will not save changes to labels...")
    # save
    # labels_df.to_csv(label_path_out + point_prefix + "_labels.csv", index=False)
    if np.all(axis_df["approved_flag"]):
        print("Saving tissue labels...")
        labels_df["tissue_model"] = seg_type_global
        labels_df.to_csv(approved_data_path + point_prefix + "_labels.csv", index=False)


# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date = "20240712_02"# "20240712_01"
    overwrite = True
    fluo_flag = False
    use_model_priors = True
    seg_model = "tdTom-bright-log-v5" #"tdTom-bright-log-v5"  # "tdTom-dim-log-v3"
    # point_model = "point_models_pos"
    well_num = 29
    time_int = 0
    curate_pec_fins(root, experiment_date=experiment_date, well_num=well_num, seg_type="tissue_only_best_model_tissue", #seg_type="seg01_best_model_tbx5a", #
                    seg_model=seg_model, time_int=time_int)



