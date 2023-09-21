# import shutil
# from ome_zarr.reader import Reader
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob2 as glob
from skimage.measure import label, regionprops, regionprops_table
# from scipy.ndimage import distance_transform_edt
from aicsimageio import AICSImage
# import math
# import dask
import os
from functions.utilities import path_leaf
# import open3d as o3d
# import fractal_tasks_core


def extract_nucleus_stats_prob(data_root, date_folder, voxel_res=1, prob_thresh=0, overwrite_flag=False):

    # get paths to raw data and processed probability stacks
    raw_root = os.path.join(data_root, "raw_data", date_folder, "")
    built_root = os.path.join(data_root, "built_data", date_folder, "")
    nucleus_root = os.path.join(data_root, "nucleus_data", date_folder, "")
    if not os.path.isdir(nucleus_root):
        os.makedirs(nucleus_root)

    # get metadata from raw file. If there arte multiple .nd2 files in single date folder, we assume that the pixel
    # sizes are the same for each
    nd2_paths = glob.glob(raw_root + "*.nd2")
    imObject = AICSImage(nd2_paths[0])

    # extract dimension info and generate a scale reference array
    pixel_res_raw = np.asarray(imObject.physical_pixel_sizes)
    im_dims = np.asarray([imObject.dims["Z"][0], imObject.dims["Y"][0], imObject.dims["X"][0]])

    z_vec = np.arange(im_dims[0]) * pixel_res_raw[0]
    y_vec = np.arange(im_dims[1]) * pixel_res_raw[1]
    x_vec = np.arange(im_dims[2]) * pixel_res_raw[2]

    z_ref_array, y_ref_array, x_ref_array = np.meshgrid(z_vec, y_vec, x_vec, indexing="ij")

    # get list of datasets with nucleus labels
    image_list = sorted(glob.glob(built_root + "*_probs.tif"))

    for im, prob_path in enumerate(tqdm(image_list)):

        # extract metadata
        prob_name = path_leaf(prob_path)
        data_name = prob_name.replace("_prob.tif", "")
        save_path = os.path.join(nucleus_root, data_name, 'morph_df.csv')

        # well number
        well_ind = prob_name.find("well")
        well_num = int(prob_name[well_ind+4:well_ind+7])

        # time step index
        time_ind = int(prob_name[well_ind + 9:well_ind+12])

        print('Extracting stats for ' + data_name)
        probImage = AICSImage(prob_path)
        prob_data = np.squeeze(probImage.data)
        # prob_data_thresh = prob_data > prob_thresh
        # nc_indices = np.where(prob_data_thresh == 1)[0]

        nc_z = z_ref_array[np.where(prob_data > prob_thresh)]
        nc_y = y_ref_array[np.where(prob_data > prob_thresh)]
        nc_x = x_ref_array[np.where(prob_data > prob_thresh)]
        

        # add layer of mask centroids
        label_array = np.asarray(label_data[level].compute())
        regions = regionprops(label_array)

        centroid_array = np.empty((len(regions), 3))
        for rgi, rg in enumerate(regions):
            centroid_array[rgi, :] = np.multiply(rg.centroid, scale_vec)

        # convert centroid array to data frame
        nucleus_df = pd.DataFrame(centroid_array, columns=["Z", "Y", "X"])

        # add additional info
        area_vec = []
        for rgi, rg in enumerate(regions):
            area_vec.append(rg.area)
        nucleus_df["Area"] = area_vec

        # calculate axis lengths
        axis_array = np.empty((len(regions), 3))
        vec_list = []
        for rgi, rg in enumerate(regions):
            moments = rg['moments_central']
            axes, axis_dirs = ellipsoid_axis_lengths(moments)
            axis_array[rgi, :] = np.multiply(axes, scale_vec)
            vec_list.append(axis_dirs)

        df2 = pd.DataFrame(axis_array, columns=["Axis_0", "Axis_1", "Axis_2"])
        df2["axis_dirs"] = vec_list

        nucleus_df = pd.concat([nucleus_df, df2], axis=1)

        if not resetClass:
            # check for class column in old dataset
            # load nucleus centroid data frame
            nucleus_df_orig = pd.read_csv(savePath, index_col=0)

            if 'pec_fin_flag' in nucleus_df_orig.columns:
                if nucleus_df.shape[0] == nucleus_df_orig.shape[0]:
                    nucleus_df['pec_fin_flag'] = nucleus_df_orig['pec_fin_flag']
                else:
                    raise Warning(
                        "Attempted to extract old pec fin assignments, but old and new datasets have different sizes. Skipping.")
            else:
                raise Warning(
                    "Attempted to extract old pec fin assignments, but 'pec_fin_flag' field does not exist. Skipping.")

        ################################
        # Calculate mRNA levels in each nucleus
        imObject = AICSImage(rawImagePath)
        image_array = imObject.data
        image_array = np.squeeze(image_array)
        # image_array = dask.array.rechunk(image_array, chunks='auto')
        # image_array = image_array.compute()

        # compute each channel of image array separately to avoid dask error
        im_array_list = []
        for ch in mRNA_channels:
            im_temp = image_array[ch, :, :, :]
            im_array_list.append(im_temp)

        # initialize empty columns
        print("Estimating cellular mRNA levels")
        omero_attrs = image_node.root.zarr.root_attrs['omero']
        channel_metadata = omero_attrs['channels']  # list of channels and relevant info
        channel_names = [channel_metadata[i]["label"] for i in mRNA_channels]
        for ch in channel_names:
            nucleus_df[ch] = np.nan
            nucleus_df[ch + "_mean"] = np.nan

        # calculate mRNA levels
        for rgi, rg in enumerate(regions):
            # iterate through channels
            nc_coords = rg.coords.astype(int)
            n_pix = nc_coords.shape[0]
            for chi, im_ch in enumerate(im_array_list):
                # nc_ch_coords = np.concatenate((np.ones((n_pix,1))*ch, nc_coords), axis=1).astype(int)
                mRNA_int = np.sum(im_ch[tuple(nc_coords.T)])
                nucleus_df.loc[rgi, channel_names[chi] + "_mean"] = mRNA_int / n_pix
                nucleus_df.loc[rgi, channel_names[chi]] = mRNA_int

        # normalize
        for ch in channel_names:
            int_max = np.max(nucleus_df[ch])
            mean_max = np.max(nucleus_df[ch + "_mean"])
            nucleus_df[ch] = nucleus_df.loc[:, ch] / int_max
            nucleus_df[ch + "_mean"] = nucleus_df.loc[:, ch + "_mean"] / mean_max

        #######################################################
        # expand nuclear masks to define cellular neighborhoods

        if overwriteCellLabels:
            label_array_cell = calculate_cell_masks(label_array, labelPath, nucleus_df, image_data, scale_vec, level,
                                                    multiscale_attrs, dataset_info)
        else:
            reader_lb_cell = Reader(parse_url(cellLabelPath))

            # nodes may include images, labels etc
            nodes_lb_cell = list(reader_lb_cell())

            # first node will be the image pixel data
            label_node_cell = nodes_lb_cell[1]
            label_array_cell = label_node_cell.data[level]

            label_array_cell = label_array_cell.compute()

        # now repeat above mRNA inference, this time using cell neighborhoods
        for ch in channel_names:
            nucleus_df[ch + "_cell"] = np.nan
            nucleus_df[ch + "_cell_mean"] = np.nan

        # calculate mRNA levels
        cell_regions = regionprops(label_array_cell)
        for rgi, rg in enumerate(cell_regions):
            # iterate through channels
            nc_coords = rg.coords.astype(int)
            n_pix = nc_coords.shape[0]
            for chi, im_ch in enumerate(im_array_list):
                # nc_ch_coords = np.concatenate((np.ones((n_pix,1))*ch, nc_coords), axis=1).astype(int)
                mRNA_int = np.sum(im_ch[tuple(nc_coords.T)])
                nucleus_df.loc[rgi, channel_names[chi] + "_cell_mean"] = mRNA_int / n_pix
                nucleus_df.loc[rgi, channel_names[chi] + "_cell"] = mRNA_int

        for ch in channel_names:
            int_max = np.max(nucleus_df.loc[:, ch + "_cell"])
            mean_max = np.max(nucleus_df.loc[:, ch + "_cell_mean"])
            nucleus_df.loc[:, ch + "_cell"] = nucleus_df.loc[:, ch + "_cell"] / int_max
            nucleus_df.loc[:, ch + "_cell_mean"] = nucleus_df.loc[:, ch + "_cell_mean"] / mean_max

        # remove empty rows (why do these appear?)
        nan_flag = np.isnan(nucleus_df.loc[:, "X"])
        nucleus_df = nucleus_df.loc[~nan_flag, :]

        # Average across nearest neighbors for both nucleus- and cell-based mRNA estimates
        nn_k = 6
        tree = KDTree(nucleus_df[["Z", "Y", "X"]], leaf_size=2)
        nearest_dist, nearest_ind = tree.query(nucleus_df[["Z", "Y", "X"]], k=nn_k + 1)

        for ch in channel_names:
            nucleus_df[ch + "_cell_nn"] = np.nan
            nucleus_df[ch + "_cell_mean_nn"] = np.nan
            nucleus_df[ch + "_nn"] = np.nan
            nucleus_df[ch + "_mean_nn"] = np.nan

        for row in range(nucleus_df.shape[0]):
            nearest_indices = nearest_ind[row, :]
            nn_df_temp = nucleus_df.iloc[nearest_indices]

            for ch in channel_names:
                int_mean = np.mean(nn_df_temp.loc[:, ch])
                mean_mean = np.mean(nn_df_temp.loc[:, ch + "_mean"])
                int_mean_cell = np.mean(nn_df_temp.loc[:, ch + "_cell"])
                mean_mean_cell = np.mean(nn_df_temp.loc[:, ch + "_cell_mean"])

                nucleus_df.loc[row, ch + "_cell_nn"] = mean_mean_cell
                nucleus_df.loc[row, ch + "_cell_mean_nn"] = int_mean_cell
                nucleus_df.loc[row, ch + "_nn"] = int_mean
                nucleus_df.loc[row, ch + "_mean_nn"] = mean_mean

        nucleus_df.to_csv(savePath)


if __name__ == "__main__":
    # define some variables
    root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/"
    date_folder = "20230830"
    prob_thresh = -8
    extract_nucleus_stats_prob(root, date_folder, prob_thresh=prob_thresh)
