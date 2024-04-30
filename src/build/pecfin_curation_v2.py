from aicsimageio import AICSImage
import numpy as np
import napari
import os
from glob2 import glob
import skimage.io as skio
from alphashape import alphashape
from functions.utilities import path_leaf
from skimage.transform import resize
import pandas as pd
# import open3d as o3d

# root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/"
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\"
date_folder = "20230913"
filename = "tdTom_prdm_pecfin_40x"
target_res_xy = 1
prob_thresh = -4

well_ind_list = [0]#, 19] #, 20]
max_time = 50
update_flag = True
# hull_alpha = 8

# load metadata
ref_image_path = os.path.join(root, "raw_data", date_folder, filename + ".nd2")
imObject = AICSImage(ref_image_path)

# git list of prob files produced by CellPose
prob_file_dir = os.path.join(root, "built_data", "cellpose_output", date_folder, "")
prob_file_list = glob(prob_file_dir + filename + "*probs*")
prob_file_list = [file for file in prob_file_list if "full_embryo" not in file]
prob_file_list = [file for file in prob_file_list if "whole_embryo" not in file]
# make directory for saving fin masks
fin_mask_dir = os.path.join(root, "fin_masks", date_folder, "")
if not os.path.isdir(fin_mask_dir):
    os.makedirs(fin_mask_dir)

# make directory for fin interior points
fin_point_dir = os.path.join(root, "fin_interior_points", date_folder, "")
if not os.path.isdir(fin_point_dir):
    os.makedirs(fin_point_dir)

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = tuple(res_raw)
scale_array = np.asarray(scale_vec)
scale_array_rs = scale_array.copy()
scale_array_rs[1::] = target_res_xy

image_i = 0
def event_trigger(event):
    if event.source.name == 'fin hull points':
        point_data = event.source.data
        calculate_fin_hull(point_data)

def calculate_fin_hull(point_data, hull_alpha=4):

    # point_data = point_data.source.data
    point_data_norm = np.divide(point_data, im_dims)
    update_lb_flag = False
    if point_data.shape[0] == 6:
        update_lb_flag = True
        
        hull = alphashape(point_data_norm, alpha=hull_alpha)
        values = np.linspace(0.5, 1, len(hull.vertices))
        surf = (np.multiply(hull.vertices, im_dims), hull.faces, values)
        fin_hull_layer = viewer.add_surface(surf, name="fin_hull", colormap="bop blue", scale=scale_vec, opacity=0.5)

    elif point_data.shape[0] > 6:
        update_lb_flag = True
        
        hull = alphashape(point_data_norm, alpha=hull_alpha)
        values = np.linspace(0.5, 1, len(hull.vertices))
        surf = (np.multiply(hull.vertices, im_dims), hull.faces, values)

        if "fin_hull" not in viewer.layers:
            fin_hull_layer = viewer.add_surface(surf, name="fin_hull", colormap="bop blue", scale=scale_vec,
                                                opacity=0.5)
        viewer.layers["fin_hull"].data = surf
        viewer.layers["fin_hull"].refresh()
    
    if update_lb_flag:
        if len(hull.faces) > 2:
            inside_flags = hull.contains(zyx_nuclei_norm)
            fin_points = zyx_nuclei[np.where(inside_flags == 1)[0], :].astype(int)
            new_labels = np.zeros(im_dims_ds, dtype=np.int8)
            new_labels[fin_points[:, 0], fin_points[:, 1], fin_points[:, 2]] = 1
            viewer.layers["labels"].data = resize(new_labels, im_dims, preserve_range=True, order=0)
            viewer.layers["labels"].refresh()

        # viewer.layers["fin points"].data = fin_points
        # viewer.layers["fin points"].refresh()

def extract_nucleus_stats_prob(prob_data, out_shape, prob_thresh=-4.0): #, voxel_res=2):

    # out_shape = np.round(np.multiply(prob_data.shape, pixel_res_vec / pixel_res_vec[0]), 0).astype(int)
    prob_data_rs = resize(prob_data, out_shape, preserve_range=True, order=1)

    z_vec = np.arange(out_shape[0]) #* pixel_res_vec[0]
    y_vec = np.arange(out_shape[1]) #* pixel_res_vec[1]
    x_vec = np.arange(out_shape[2]) #* pixel_res_vec[2]

    z_ref_array, y_ref_array, x_ref_array = np.meshgrid(z_vec, y_vec, x_vec, indexing="ij")

    # extract 3D positions of each foreground pixel. This is just a high-res point cloud
    nc_z = z_ref_array[np.where(prob_data_rs > prob_thresh)]
    nc_y = y_ref_array[np.where(prob_data_rs > prob_thresh)]
    nc_x = x_ref_array[np.where(prob_data_rs > prob_thresh)]

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    zyx_out = np.concatenate((nc_z[:, np.newaxis], nc_y[:, np.newaxis], nc_x[:, np.newaxis]), axis=1)

    return zyx_out

global z_ref_array, y_ref_array, x_ref_array, im_dims, xyz_array

# determine how many unique embryos and time points we have
well_id_vec = []
time_id_vec = []
for p in range(len(prob_file_list)):
    prob_name = prob_file_list[p]
    prob_name_short = path_leaf(prob_name)

    # well number
    well_ind = prob_name.find("well")
    well_num = int(prob_name[well_ind + 4:well_ind + 7])
    well_id_vec.append(well_num)

    # time step index
    time_ind = int(prob_name[well_ind + 9:well_ind + 12])
    time_id_vec.append(time_ind)

metadata_array = np.concatenate((np.asarray(well_id_vec)[:, np.newaxis], np.asarray(time_id_vec)[:, np.newaxis]), axis=1)
keep_indices = np.asarray([i for i in range(metadata_array.shape[0]) if well_id_vec[i] in well_ind_list])
metadata_array = metadata_array[keep_indices, :]


prob_file_list_sorted = np.asarray(prob_file_list)
prob_file_list_sorted = prob_file_list_sorted[keep_indices]
prob_file_list_sorted = prob_file_list_sorted[np.lexsort((np.asarray(metadata_array[:, 1]), metadata_array[:, 0]))][::-1]

metadata_array = metadata_array[np.lexsort((np.asarray(metadata_array[:, 1]), metadata_array[:, 0])), :][::-1]
t_indices = np.where(metadata_array[:, 1] <= max_time)[0]

prob_file_list_sorted = prob_file_list_sorted[t_indices]
prev_well = None

while image_i < len(prob_file_list_sorted):
    prob_name = prob_file_list_sorted[image_i]

    im_prob = skio.imread(prob_name, plugin="tifffile")
    im_bin = im_prob >= prob_thresh

    ############
    # check for existing points file
    prob_name_short = path_leaf(prob_name)
    point_name = prob_name_short.replace("probs.tif", "fin_mask_points.npy")
    point_path = os.path.join(fin_mask_dir, point_name)

    # well number
    well_ind = prob_name.find("well")
    well_num = int(prob_name[well_ind + 4:well_ind + 7])
    carry_flag = False
    if well_num == prev_well:
        carry_flag = True

    # time step index
    time_ind = int(prob_name[well_ind + 9:well_ind + 12])

    nucleus_name = prob_name_short.replace("probs.tif", "fin_interior_points.csv")
    nucleus_path = os.path.join(fin_point_dir, nucleus_name)
    skip_flag = False
    if os.path.isfile(nucleus_path) and (not update_flag):
        skip_flag = True

        point_array = np.load(point_path)
        point_array = np.divide(point_array, scale_vec)

    # if time_ind > 22:
    #     skip_flag = True
    #     print("Skipping temporarily")

    if not skip_flag:
        ############
        # generate reference arrays
        im_dims = im_prob.shape
        ds_array = np.divide(scale_array_rs, scale_array)
        im_dims_ds = np.round(np.divide(np.asarray(im_dims), ds_array)).astype(int)

        z_vec = np.arange(im_dims_ds[0])
        y_vec = np.arange(im_dims_ds[1])
        x_vec = np.arange(im_dims_ds[2])

        z_ref_array, y_ref_array, x_ref_array = np.meshgrid(z_vec, y_vec, x_vec, indexing="ij")
        zyx_array = np.concatenate((z_ref_array.flatten()[:, np.newaxis],
                                    y_ref_array.flatten()[:, np.newaxis],
                                    x_ref_array.flatten()[:, np.newaxis]), axis=1)
        zyx_array_norm = np.divide(zyx_array, im_dims_ds)

        #############
        # get positions of nuclei
        zyx_nuclei = extract_nucleus_stats_prob(im_prob, im_dims_ds)
        zyx_nuclei_norm = np.divide(zyx_nuclei, im_dims_ds)

        # initialize viewer
        viewer = napari.view_image(im_prob, colormap="gray", scale=scale_vec,
                                   contrast_limits=(prob_thresh, np.percentile(im_prob, 99.5)))

        #############
        # generate label array
        label_array = np.zeros(im_dims, dtype=np.uint8)
        label_layer = viewer.add_labels(label_array, name='labels', scale=scale_vec)

        # initialize points layer
        if os.path.isfile(point_path):
            point_array = np.load(point_path)
            point_array = np.divide(point_array, scale_vec)
            # points_layer = viewer.add_points(point_array, name="fin hull points", size=8, scale=scale_vec,
            #                                  n_dimensional=True)
            if point_array.shape[0] >= 6:
                calculate_fin_hull(point_array)
        elif carry_flag:
            point_array = point_array_prev
        else:
            point_array = np.empty((0, 3))
        points_layer = viewer.add_points(point_array, name="fin hull points", size=8, scale=scale_vec, n_dimensional=True)

        points_layer.events.data.connect(event_trigger)

        napari.run()

        points_layer = viewer.layers["fin hull points"]
        label_array = np.asarray(viewer.layers["labels"].data)

        # save hull points to file
        hull_point_array = np.asarray(points_layer.data)
        hull_point_array = np.multiply(hull_point_array, scale_array)
        np.save(point_path, hull_point_array)

        # save fin points to file
        fin_point_array = extract_nucleus_stats_prob(label_array, im_dims_ds, prob_thresh=0.5)
        fin_point_array = np.multiply(fin_point_array, scale_array_rs)
        fin_df = pd.DataFrame(fin_point_array, columns=["Z", "Y", "X"])

        # fin_df.loc[:, "filename"] = prob_name_short
        fin_df.loc[:, "well_num"] = well_num
        fin_df.loc[:, "time_ind"] = time_ind
        fin_df.loc[:, "xy_res"] = scale_array_rs[1]
        fin_df.loc[:, "z_res"] = scale_array_rs[0]
        fin_df.loc[:, "fin_flag"] = 1

        # save other points to file
        prob_array_other = im_prob.copy()
        prob_array_other[label_array > 0.5] = -100  # some large negative number
        other_point_array = extract_nucleus_stats_prob(prob_array_other, im_dims_ds, prob_thresh=prob_thresh)
        other_point_array = np.multiply(other_point_array, scale_array_rs)
        other_df = pd.DataFrame(other_point_array, columns=["Z", "Y", "X"])

        # fin_df.loc[:, "filename"] = prob_name_short
        other_df.loc[:, "well_num"] = well_num
        other_df.loc[:, "time_ind"] = time_ind
        other_df.loc[:, "xy_res"] = scale_array_rs[1]
        other_df.loc[:, "z_res"] = scale_array_rs[0]
        other_df.loc[:, "fin_flag"] = 0

        # viewer.add_points(other_point_array, name="other points", size=8, n_dimensional=True)

        fin_label_df = pd.concat([fin_df, other_df], axis=0, ignore_index=True)

        fin_label_df.to_csv(nucleus_path)
    else:
        print("Existing annotated nucleus file found. Skipping...")
    # update iterator
    point_array_prev = point_array
    prev_well = well_num

    wait = input(
        "Press Enter to continue to next image. \nPress 'x' then Enter to exit. \nPress Enter to move to next experiment.")
    if wait == 'x':
        break
    # elif wait == 'n':
    #     image_i += 1
    else:
        image_i += 1
        # print(image_i)
# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
# if __name__ == '__main__':
#     napari.run()