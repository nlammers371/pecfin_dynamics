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
import nd2
import zarr
# from ultrack.utils import estimate_parameters_from_labels, labels_to_edges

# set paths
root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"

experiment_date = "20240223"
model = "tdTom-v4-20240315"
data_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')
label_directory = os.path.join(root, "built_data", "cellpose_output", model, experiment_date, '')

# get list of images
image_list = sorted(glob.glob(data_directory + "*.zarr"))

# zarr_path = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
time_int = 0
well_int = 12

scale_vec = tuple([2.0, 0.55, 0.55])

# prob_name = experiment_date + f"_well{well_int:03}_t{time_int:03}_probs.tif"
# prob_path = os.path.join(label_directory, prob_name)
# im_prob = io.imread(prob_path)
# im_prob = im_prob[np.newaxis, :, :, :]

# extract key metadata info
data_tzyx_full = zarr.open(image_list[well_int], mode="r")
data_tzyx = np.squeeze(data_tzyx_full[time_int])


# estimate background using blur
top1 = np.percentile(data_tzyx, 99)
data_capped = data_tzyx.copy()
data_capped[data_capped > top1] = top1
data_capped = data_capped[:, 500:775, 130:475]

shape_orig = np.asarray(data_capped.shape)
shape_iso = shape_orig.copy()
iso_factor = scale_vec[0] / scale_vec[1]
shape_iso[0] = shape_iso[0] * iso_factor

gaussian_background = ski.filters.gaussian(data_capped, sigma=(2, 8, 8))
data_1 = np.divide(data_capped, gaussian_background)
# data_sobel = ski.filters.sobel(data_1)
# data_sobel_i = ski.util.invert(data_sobel)

data_rs = resize(data_1, shape_iso, preserve_range=True, order=1)
image = sitk.GetImageFromArray(data_rs)
data_log = sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(image, sigma=1))
# data_log_i = resize(ski.util.invert(data_log), shape_orig, preserve_range=True, order=1)
data_log_i = ski.util.invert(data_log)

# rescale and convert to 16 bit
data_rs_16 = data_rs.copy()
data_rs_16 = data_rs_16 - np.min(data_rs_16)
data_rs_16 = np.round(data_rs_16 / np.max(data_rs_16) * 2**16 - 1).astype(np.uint16)

log_i_16 = data_log_i.copy()
log_i_16 = log_i_16 - np.min(log_i_16)
log_i_16 = np.round(log_i_16 / np.max(log_i_16) * 2**16 - 1).astype(np.uint16)

viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(data_capped, scale=scale_vec)
viewer.add_image(data_rs_16, name="background removed")
# viewer.add_image(data_sobel, scale=scale_vec, name="sobel")
# viewer.add_image(data_sobel_i, scale=scale_vec, name="sobel-inverted")
# viewer.add_image(data_log, name="log")
viewer.add_image(log_i_16, name="log-inverted")


train_path_log = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\built_data\\cellpose_training\\20240223_tdTom\\log\\"
train_path_bk = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\built_data\\cellpose_training\\20240223_tdTom\\bk\\"
if not os.path.isdir(train_path_bk):
    os.makedirs(train_path_bk)
if not os.path.isdir(train_path_log):
    os.makedirs(train_path_log)

z_slice_vec = [83, 100, 104, 115]
x_slice_vec = [105, 116, 143]
y_slice_vec = [176]

for i in range(len(z_slice_vec)):
    z_slice_log = np.squeeze(data_log_i[z_slice_vec[i], :, :])
    io.imsave(os.path.join(train_path_log, f"xy_slice{i:03}.tiff"), z_slice_log, check_contrast=False)
    z_slice_bk = np.squeeze(data_rs[z_slice_vec[i], :, :])
    io.imsave(os.path.join(train_path_bk, f"xy_slice{i:03}.tiff"), z_slice_bk, check_contrast=False)

for i in range(len(x_slice_vec)):
    x_slice_log = np.squeeze(data_log_i[:, :, x_slice_vec[i]])
    io.imsave(os.path.join(train_path_log, f"zy_slice{i:03}.tiff"), x_slice_log, check_contrast=False)
    x_slice_bk = np.squeeze(data_rs[:, :, x_slice_vec[i]])
    io.imsave(os.path.join(train_path_bk, f"zy_slice{i:03}.tiff"), x_slice_bk, check_contrast=False)

for i in range(len(y_slice_vec)):
    y_slice_log = np.squeeze(data_log_i[:, y_slice_vec[i], :])
    io.imsave(os.path.join(train_path_log, f"zx_slice{i:03}.tiff"), y_slice_log, check_contrast=False)
    y_slice_bk = np.squeeze(data_rs[:, y_slice_vec[i], :])
    io.imsave(os.path.join(train_path_bk, f"zx_slice{i:03}.tiff"), y_slice_bk, check_contrast=False)

if __name__ == '__main__':
    napari.run()
# viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(label(detection[0]), scale=scale_vec)
# viewer.add_labels(label(boundaries[0]), scale=scale_vec)