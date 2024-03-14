import napari
import os
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from ultrack.imgproc.intensity import robust_invert
from ultrack.imgproc.segmentation import detect_foreground
import time
from ultrack.utils.array import array_apply
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import nd2
import zarr
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
import glob2 as glob

# get path to raw nd2 file


def perform_tracking(nd2_path, root, experiment_date, well_num, tracking_config, start_i=0, stop_i=None,
                     overwrite_flag=False):

    imObject = nd2.ND2File(nd2_path)
    res_raw = imObject.voxel_size()
    scale_vec = np.asarray(res_raw)[::-1]

    # set parameters

    data_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, "")

    tracking_name = tracking_config.replace(".txt", "")
    # get path to zarr file
    filename = experiment_date + f"_well{well_num:04}.zarr"
    zarr_path = os.path.join(data_path, filename)
    data_tzyx_raw = zarr.open(zarr_path, mode='r')
    if stop_i is None:
        stop_i = data_tzyx_raw.shape[0]
    # get path to metadata
    metadata_path = os.path.join(root, "metadata", "tracking")

    # set output path for tracking results
    project_path = os.path.join(root, "tracking", experiment_date,  tracking_name, f"well{well_num:04}", "")
    if not os.path.isdir(project_path):
        os.makedirs(project_path)

    # specify time points to load
    data_tzyx = data_tzyx_raw[start_i:stop_i, :, 300:700, :]
    # n_time_points = zarr_file.shape[0]

    # load tracking config file
    cfg = load_config(os.path.join(metadata_path, tracking_config))
    cfg.data_config.working_dir = project_path
    # print(cfg)

    # f = open(os.path.join(project_path, tracking_config), "w")
    # f.write("Now the file has more content!")
    # f.close()

    d_zarr_path = os.path.join(project_path, "detection.zarr")
    if os.path.isdir(d_zarr_path):
        mask_data = zarr.open(d_zarr_path, mode='a', shape=data_tzyx.shape, dtype=np.uint16,
                           chunks=(1,) + data_tzyx.shape[1:])

        detection, boundaries = labels_to_edges(mask_data)
    else:
        # segment
        print("Performing segmentation...")
        # start = time.time()
        detection = np.empty(data_tzyx.shape, dtype=np.uint)
        array_apply(
            data_tzyx,
            out_array=detection,
            func=detect_foreground,
            sigma=15.0,
            voxel_size=scale_vec,
        )

        print("Calculating boundaries...")
        boundaries = np.empty(data_tzyx.shape, dtype=np.uint)
        array_apply(
            data_tzyx,
            out_array=boundaries,
            func=robust_invert,
            voxel_size=scale_vec,
        )

        # save detection and boundary files to zarr
        d_zarr = zarr.open(d_zarr_path, mode='a', shape=data_tzyx.shape, dtype=np.uint16, chunks=(1,) + data_tzyx.shape[1:])
        d_zarr[:] = detection

# print("Examine segmentation in napari")
# viewer = napari.view_image(data_tzyx, scale=tuple(scale_vec))
# label_layer = viewer.add_labels(detection, name='segmentation', scale=tuple(scale_vec))
# boundary_layer = viewer.add_image(boundaries, visible=False, scale=tuple(scale_vec))
# viewer.theme = "dark"

    # Perform tracking
    print("Performing tracking...")
    track(
        cfg,
        detection=detection,
        edges=boundaries,
        scale=scale_vec,
        overwrite=True,
    )

    print("Saving results...")
    tracks_df, graph = to_tracks_layer(cfg)
    tracks_df.to_csv(project_path + "tracks.csv", index=False)

    segments = tracks_to_zarr(
        cfg,
        tracks_df,
        store_or_path=project_path + "segments.zarr",
        overwrite=True,
    )
    print("Done.")

    return segments


if __name__ == '__main__':

    nd2_path = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
    experiment_date = "20240223"
    root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
    overwrite_flag = False
    well_num = 2
    n_time_points = 200
    tracking_config = "tracking_v1.txt"

    perform_tracking(nd2_path, root, experiment_date, well_num, tracking_config, start_i=0, stop_i=30, overwrite_flag=overwrite_flag)
# print("Saving downsampled image data...")
# np.save(project_path + "image_data_ds.npy", data_tzyx)