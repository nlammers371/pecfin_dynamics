from tqdm import tqdm
import napari
import glob2 as glob
import os
import skimage.io as io
import numpy as np
import SimpleITK as sitk
# import pyclesperanto as cle
import skimage as ski
from skimage.transform import resize
from src.utilities.image_utils import calculate_LoG
import zarr
# from ultrack.utils import estimate_parameters_from_labels, labels_to_edges

# set paths
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"

experiment_date = "20240424"
model = "log-v2"
data_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')
label_directory = os.path.join(root, "built_data", "cellpose_output", model, experiment_date, '')
if not os.path.isdir(label_directory):
    os.makedirs(label_directory)

# get list of images
image_list = sorted(glob.glob(data_directory + "*.zarr"))

# zarr_path = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
time_int = 20
well_int = 12

scale_vec = [2.0, 0.55, 0.55]

# prob_name = experiment_date + f"_well{well_int:03}_t{time_int:03}_probs.tif"
# prob_path = os.path.join(label_directory, prob_name)
# im_prob = io.imread(prob_path)
# im_prob = im_prob[np.newaxis, :, :, :]

# extract key metadata info
data_tzyx_full = zarr.open(image_list[well_int], mode="r")
data_zyx = np.squeeze(data_tzyx_full[1, time_int])

im_log, im_bkg = calculate_LoG(data_zyx=data_zyx, scale_vec=scale_vec, make_isotropic=True)


viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(data_capped, scale=scale_vec)
viewer.add_image(im_log, name="background removed")



train_path_log = os.path.join(root, "built_data/cellpose_training/20240424_tdTom/log/")
train_path_bk = os.path.join(root, "built_data/cellpose_training/20240424_tdTom/bkg/")
if not os.path.isdir(train_path_bk):
    os.makedirs(train_path_bk)
if not os.path.isdir(train_path_log):
    os.makedirs(train_path_log)

z_slice_vec = [20, 40, 60, 83, 100, 104, 115]
x_slice_vec = [516, 490, 530]
y_slice_vec = [220, 230, 240]

for i in range(len(z_slice_vec)):
    z_slice_log = np.squeeze(im_log[z_slice_vec[i], :, :])
    io.imsave(os.path.join(train_path_log, f"xy_slice{i:03}.tiff"), z_slice_log, check_contrast=False)
    z_slice_bk = np.squeeze(im_bkg[z_slice_vec[i], :, :])
    io.imsave(os.path.join(train_path_bk, f"xy_slice{i:03}.tiff"), z_slice_bk, check_contrast=False)

for i in range(len(x_slice_vec)):
    x_slice_log = np.squeeze(im_log[:, :, x_slice_vec[i]])
    io.imsave(os.path.join(train_path_log, f"zy_slice{i:03}.tiff"), x_slice_log, check_contrast=False)
    x_slice_bk = np.squeeze(im_bkg[:, :, x_slice_vec[i]])
    io.imsave(os.path.join(train_path_bk, f"zy_slice{i:03}.tiff"), x_slice_bk, check_contrast=False)

for i in range(len(y_slice_vec)):
    y_slice_log = np.squeeze(im_log[:, y_slice_vec[i], :])
    io.imsave(os.path.join(train_path_log, f"zx_slice{i:03}.tiff"), y_slice_log, check_contrast=False)
    y_slice_bk = np.squeeze(im_bkg[:, y_slice_vec[i], :])
    io.imsave(os.path.join(train_path_bk, f"zx_slice{i:03}.tiff"), y_slice_bk, check_contrast=False)

if __name__ == '__main__':
    napari.run()
# viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(label(detection[0]), scale=scale_vec)
# viewer.add_labels(label(boundaries[0]), scale=scale_vec)