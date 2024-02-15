# from aicsimageio import AICSImage
import numpy as np
import napari
from skimage.transform import resize
import glob2 as glob
import os
import nd2
from dask_image.imread import imread
import dask.array as da
import skimage.io as io
from tqdm import tqdm
# read the image data
from ome_zarr.io import parse_url

# set paths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
experiment_date = "20231214"
start_id = 30
stop_id = 60

# read in raw datafile
nd2_list = glob.glob(os.path.join(root, "raw_data", experiment_date, "*.nd2"))
nd2_path = nd2_list[0]
# read the image data
imObject = nd2.ND2File(nd2_path)
im_raw_dask = imObject.to_dask()
shape_raw = im_raw_dask.shape
# im_raw_dask = da.reshape(im_raw_dask, (shape_raw[0]*shape_raw[1], shape_raw[2], shape_raw[3], shape_raw[4]))
scale_vec = imObject.voxel_size()
# scale_vec_plot = tuple([1, 1]) + scale_vec[::-1]

scale_vec_plot = tuple([1, 1]) + scale_vec[::-1]
# read in label files
prob_dir = os.path.join(root, "built_data", "cellpose_output", experiment_date, "")

# im_prob_dask = imread(prob_dir + "*_probs.tif")
# im_prob_dask = da.reshape(im_prob_dask, shape_raw)
# im_prob_dask = da.moveaxis(im_prob_dask, 1, 0)  # move well index to be on the outside
# # viewer = napari.view_image(im_raw_dask, scale=scale_vec_plot)
# viewer = napari.view_image(im_prob_dask,  scale=scale_vec_plot, contrast_limits=[-8, 16])
# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)

print("Loading prob data stacks")
stop_id = np.min([stop_id, shape_raw[1]])
prob_stack = np.empty((stop_id-start_id, shape_raw[0], shape_raw[2], shape_raw[3], shape_raw[4]), dtype=np.float32)
iter_i = 0
for series_id in tqdm(range(start_id, stop_id)):

    prob_files = sorted(glob.glob(prob_dir + f"*well{series_id:03}*_probs.tif"))
    prob_list = []
    for pf in prob_files:
        im_temp = io.imread(pf)
        prob_list.append(im_temp[np.newaxis, :, :, :])

    prob_stack_temp = np.concatenate(tuple(prob_list), axis=0)

    prob_stack[iter_i, :, :, :, :] = prob_stack_temp
    iter_i += 1


viewer = napari.view_image(prob_stack, scale=scale_vec_plot, contrast_limits=[-8, 32])

napari.run()
    # print(f"Current series ID: {series_id}")

# if __name__ == '__main__':
#     napari.run()