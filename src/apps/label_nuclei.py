import os
import numpy as np
import pandas as pd
import napari
import zarr

from skimage.measure import regionprops_table
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


# -----------------------------
# GLOBAL STATE
# -----------------------------
viewer = None
label_layer = None
image_layer = None
mask_layer = None
mask_zarr = None
points_df = None
current_t = 0
time_indices = []
scale_vec = (1, 1, 1)
root_global = ""
experiment_date_global = ""
seg_model_global = ""
save_dir_global = ""
zarr_file_prefix_global = ""


# -----------------------------
# BASIC I/O
# -----------------------------
def load_zarr_data(root, experiment_date, seg_model, file_prefix):
    """
    Minimal loader for image and mask zarr datasets.
    Expects:
        image: root/built_data/zarr_image_files/{date}/{prefix}.zarr
        mask:  root/built_data/mask_stacks/{seg_model}/{date}/{prefix}_mask_aff.zarr
    """
    raw_zarr_path = os.path.join(
        root, "built_data", "zarr_image_files", experiment_date, file_prefix + ".zarr"
    )
    mask_zarr_path = os.path.join(
        root,
        "built_data",
        "mask_stacks",
        seg_model,
        experiment_date,
        file_prefix + "_mask_aff.zarr",
    )

    data_zarr = zarr.open(raw_zarr_path, mode="r")
    mask_zarr = zarr.open(mask_zarr_path, mode="r")

    scale_vec = tuple(data_zarr.attrs.get("voxel_size_um", (1, 1, 1)))

    return data_zarr, mask_zarr, scale_vec


def ensure_save_dir(root, experiment_date):
    out_dir = os.path.join(root, "point_cloud_data", "manual_labels", experiment_date)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_labels(points_df, well_num, time_int):
    global save_dir_global, experiment_date_global
    out_name = f"{experiment_date_global}_well{well_num:04}_time{time_int:04}_labels.csv"
    out_path = os.path.join(save_dir_global, out_name)
    points_df.to_csv(out_path, index=False)
    print(f"[save] wrote labels → {out_path}")


# -----------------------------
# REGIONPROP FEATURE EXTRACTION
# -----------------------------
def extract_point_features(mask, image):
    """
    Uses regionprops_table to extract centroid coordinates and intensity statistics
    directly from the segmentation mask.
    Returns a DataFrame with columns:
        nucleus_id, X, Y, Z, volume, intensity_mean, intensity_max, intensity_min
    """
    props = regionprops_table(
        mask,
        intensity_image=image,
        properties=(
            "label",
            "centroid",
            "area",
            "intensity_mean",
            "intensity_max",
            "intensity_min",
        ),
    )

    df = pd.DataFrame(props)
    df.rename(
        columns={
            "label": "nucleus_id",
            "centroid-0": "Z",
            "centroid-1": "Y",
            "centroid-2": "X",
            "area": "volume",
        },
        inplace=True,
    )
    df["fin_label_curr"] = 0
    return df


# -----------------------------
# FEATURE + MLP HELPERS
# -----------------------------
def make_placeholder_features(df):
    """
    Builds a minimal numeric feature matrix from geometric and intensity columns.
    """
    cols = ["X", "Y", "Z", "volume", "intensity_mean", "intensity_max", "intensity_min"]
    cols = [c for c in cols if c in df.columns]
    X = df[cols].to_numpy().astype(float)
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True) + 1e-6
    X_poly = np.column_stack(
        [X, X[:, 0] * X[:, 1], X[:, 0] * X[:, 2], X[:, 1] * X[:, 2]]
    )
    pca = PCA(n_components=min(6, X_poly.shape[1]))
    return pca.fit_transform(X_poly)


def run_inference(points_df):
    """
    Train a tiny MLP on labeled points (label != 0)
    and predict for unlabeled ones.
    """
    feats = make_placeholder_features(points_df)
    labels = points_df["fin_label_curr"].to_numpy()
    mask_labeled = labels != 0

    if mask_labeled.sum() < 5:
        print("[mlp] not enough labeled points to train; skipping.")
        return points_df

    X_train = feats[mask_labeled]
    y_train = labels[mask_labeled]
    X_pred = feats[~mask_labeled]

    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=0)
    clf.fit(X_train, y_train)

    if len(X_pred) > 0:
        y_pred = clf.predict(X_pred)
        labels[~mask_labeled] = y_pred
        points_df["fin_label_curr"] = labels
        print(f"[mlp] filled {len(X_pred)} unlabeled points.")

    return points_df


# -----------------------------
# NAPARI CALLBACKS
# -----------------------------
def label_update_function(event):
    """Propagate brush label to all nuclei touched by the paint stroke."""
    global label_layer, mask_zarr, points_df

    if event.type != "paint":
        return

    # ---- Gather painted voxel coordinates ----
    # event.value is a list of (coords, values) tuples per brush fragment
    z_all, y_all, x_all = [], [], []
    for stroke in event.value:
        z_all.append(stroke[0][0])
        y_all.append(stroke[0][1])
        x_all.append(stroke[0][2])
    if len(z_all) == 0:
        return
    z_all = np.concatenate(z_all)
    y_all = np.concatenate(y_all)
    x_all = np.concatenate(x_all)

    # ---- Identify nuclei IDs under the brush ----
    nucleus_ids = mask_zarr[z_all, y_all, x_all]
    nucleus_ids = nucleus_ids[nucleus_ids != 0]
    touched_ids = np.unique(nucleus_ids)
    if len(touched_ids) == 0:
        return

    # ---- Determine the label being painted with ----
    new_label = label_layer.selected_label

    # ---- Update entire nuclei regions ----
    data = label_layer.data.copy()
    data[mask_zarr == 0] = 0  # ensure background is zero
    for nid in touched_ids:
        data[mask_zarr == nid] = new_label
        points_df.loc[points_df["nucleus_id"] == nid, "fin_label_curr"] = new_label
    label_layer.data = data  # re-assign so napari redraws

    # ---- Console feedback ----
    print(f"[paint] updated {len(touched_ids)} nuclei → label {new_label}")


def save_keybinding(viewer):
    global points_df, current_t
    well_num = viewer.layers[0].metadata.get("well_num", 0)
    save_labels(points_df, well_num=well_num, time_int=current_t)


def inference_keybinding(viewer):
    global points_df, label_layer, mask_zarr
    points_df = run_inference(points_df)
    for nid, lbl in zip(points_df["nucleus_id"], points_df["fin_label_curr"]):
        if lbl != 0:
            label_layer.data[mask_zarr == nid] = lbl
    label_layer.data[mask_zarr == 0] = 0


def prev_frame(viewer):
    global current_t, time_indices
    current_t = (current_t - 1) % len(time_indices)
    viewer.close()


def next_frame(viewer):
    global current_t, time_indices
    current_t = (current_t + 1) % len(time_indices)
    viewer.close()


# -----------------------------
# MAIN APP LOOP
# -----------------------------
def run_label_app(root, experiment_date, seg_model, well_num):
    """
    Main loop:
        - loads mask & image zarrs
        - extracts features directly
        - displays interactive napari app
    """
    global viewer, label_layer, mask_layer, image_layer
    global mask_zarr, points_df, current_t, time_indices
    global root_global, experiment_date_global, seg_model_global, save_dir_global

    root_global = root
    experiment_date_global = experiment_date
    seg_model_global = seg_model
    save_dir_global = ensure_save_dir(root, experiment_date)

    file_prefix = f"{experiment_date}_well{well_num:04}"
    data_zarr, mask_zarr_all, scale = load_zarr_data(root, experiment_date, seg_model, file_prefix)

    n_time = len(mask_zarr_all)
    time_indices = list(range(n_time))
    current_t = 0

    while True:
        print(f"[loop] loading time index {current_t} / {n_time-1}")
        mask = mask_zarr_all[current_t]

        # load image slice
        if data_zarr.ndim == 4:
            im_t = data_zarr[current_t]
        elif data_zarr.ndim == 5:
            im_t = data_zarr[current_t, 0]
        else:
            raise ValueError("Unexpected zarr shape")

        # compute regionprops
        points_df_curr = extract_point_features(mask, im_t)

        mask_zarr = mask
        points_df = points_df_curr
        scale_vec = scale

        # init label volume
        label_vol = np.zeros_like(mask, dtype=np.int32)

        # launch viewer
        viewer = napari.Viewer(ndisplay=3)
        im_t = np.asarray(im_t, dtype=np.float32)
        image_layer = viewer.add_image(im_t, name="image", scale=scale_vec)
        image_layer.metadata["well_num"] = well_num
        mask_layer = viewer.add_labels(mask, name="nuclei masks", scale=scale_vec, opacity=0.25)
        label_layer = viewer.add_labels(label_vol, name="labels", scale=scale_vec, opacity=0.65)
        label_layer.brush_size = 20
        label_layer.events.paint.connect(label_update_function)

        viewer.bind_key("s", save_keybinding)
        viewer.bind_key("p", inference_keybinding)
        viewer.bind_key(",", prev_frame)
        viewer.bind_key(".", next_frame)

        napari.run()

        save_labels(points_df, well_num=well_num, time_int=current_t)


if __name__ == "__main__":
    root = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date = "20240619"
    seg_model = "tdTom-bright-log-v5"
    well_num = 2

    run_label_app(root=root, experiment_date=experiment_date, seg_model=seg_model, well_num=well_num)
