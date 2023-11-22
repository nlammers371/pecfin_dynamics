from aicsimageio import AICSImage
import numpy as np
import napari
import os
from glob2 import glob
import skimage.io as skio
from alphashape import alphashape
from functions.utilities import path_leaf
from skimage.transform import resize
# read the image data
from ome_zarr.io import parse_url

# full_filename = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\pec_fin_dynamics\\fin_morphodynamics\\raw_data\\20230830\\tdTom_54hpf_pecfin_40x.nd2"
root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/"
date_folder = "20230830"
filename = "tdTom_54hpf_pecfin_40x"

# load metadata
ref_image_path = os.path.join(root, "raw_data", date_folder, filename + ".nd2")
imObject = AICSImage(ref_image_path)

# git list of prob files produced by CellPose
prob_file_dir = os.path.join(root, "built_data", date_folder, "")
prob_file_list = glob(prob_file_dir + filename + "*probs*")

# make directory for saving fin masks
fin_mask_dir = os.path.join(root, "fin_masks", date_folder, "")
if not os.path.isdir(fin_mask_dir):
    os.makedirs(fin_mask_dir)

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = tuple(res_raw)

image_i = 0
prob_thresh = -4
scale_array = np.asarray(scale_vec)
label_ds_factor = 4
def calculate_fin_hull(event):
    # print(f"{event.source.name} changed its data!")
    point_data = event.source.data
    point_data_norm = np.divide(point_data, im_dims)
    update_lb_flag = False
    if point_data.shape[0] == 6:
        update_lb_flag = True
        
        hull = alphashape(point_data_norm, alpha=0.01)
        values = np.linspace(0.5, 1, len(hull.vertices))
        surf = (np.multiply(hull.vertices, im_dims), hull.faces, values)
        fin_hull_layer = viewer.add_surface(surf, name="fin_hull", colormap="bop blue", scale=scale_vec, opacity=0.5)
        # fin_hull_layer.editable = True
        # viewer.add_points(np.asarray([3, 100, 100]), name="test", size=15)
    elif point_data.shape[0] > 6:
        update_lb_flag = True
        
        hull = alphashape(point_data_norm, alpha=0.01)
        values = np.linspace(0.5, 1, len(hull.vertices))
        surf = (np.multiply(hull.vertices, im_dims), hull.faces, values)
        viewer.layers["fin_hull"].data = surf
        viewer.layers["fin_hull"].refresh()
    
    if update_lb_flag:
        # lb_data = viewer.layers["labels"].data
        inside_flags = hull.contains(zyx_array)
        inside_array = np.reshape(inside_flags, (im_dims_ds))
        lb_rescale = resize(inside_array.astype(float), im_dims, preserve_range=False, order=1) >= 0.5
        viewer.layers["labels"].data = lb_rescale#np.multiply(resize(inside_array, im_dims, preserve_range=True, order=0), im_bin)
        viewer.layers["labels"].refresh()

global z_ref_array, y_ref_array, x_ref_array, im_dims, xyz_array

while image_i < len(prob_file_list):
    prob_name = prob_file_list[image_i]

    im_prob = skio.imread(prob_name, plugin="tifffile")
    im_bin = im_prob >= prob_thresh
    viewer = napari.view_image(im_prob, colormap="gray", scale=scale_vec, contrast_limits=(prob_thresh, np.percentile(im_prob, 99.5)))

    ############
    # check for existing points file
    prob_name_short = path_leaf(prob_name)
    point_name = prob_name_short.replace("probs.tif", "fin_mask_points.npy")
    point_path = os.path.join(fin_mask_dir, point_name)
    if os.path.isfile(point_path):
        point_array = np.load(point_path)
    else:
        point_array = np.empty((0, 3))
    points_layer = viewer.add_points(point_array, name="fin hull points", size=8, scale=scale_vec, n_dimensional=True)

    ############
    # generate reference arrays
    im_dims = im_prob.shape
    im_dims_ds = np.floor(np.asarray(im_dims) / label_ds_factor).astype(int)
    scale_vec_lb = tuple(np.divide(np.multiply(scale_array, im_dims), im_dims_ds))
    z_vec = np.arange(im_dims_ds[0])
    y_vec = np.arange(im_dims_ds[1])
    x_vec = np.arange(im_dims_ds[2])

    z_ref_array, y_ref_array, x_ref_array = np.meshgrid(z_vec, y_vec, x_vec, indexing="ij")
    zyx_array = np.concatenate((z_ref_array.flatten()[:, np.newaxis],
                                y_ref_array.flatten()[:, np.newaxis],
                                x_ref_array.flatten()[:, np.newaxis]), axis=1)
    zyx_array = np.divide(zyx_array, im_dims_ds)
    #############
    # generate label array
    label_array = np.zeros(im_dims, dtype=np.uint8)
    label_layer = viewer.add_labels(label_array, name='labels', scale=scale_vec)

    points_layer.events.data.connect(calculate_fin_hull)

    napari.run()

    points_layer = viewer.layers["fin hull points"]

    # save to file
    point_array = np.asarray(points_layer.data)
    point_array = np.multiply(point_array, scale_vec)


    np.save(point_path, point_array)

    wait = input(
        "Press Enter to continue to next image. \nPress 'x' then Enter to exit. \nPress 'n' then Enter to move to next experiment.")
    if wait == 'x':
        break
    elif wait == 'n':
        break
    else:
        image_i += 1
        print(image_i)
# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
# if __name__ == '__main__':
#     napari.run()