from aicsimageio import AICSImage
import numpy as np
import napari
from skimage.transform import resize
import glob2 as glob
import os
# read the image data
import nd2
from tqdm import tqdm

# root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
# experiment_date = "20231013"

# nd2_list = glob.glob(os.path.join())
# full_filename = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/raw_data/20230830/tdTom_54hpf_pecfin_40x.nd2"
full_filename = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
imObject = nd2.ND2File(full_filename)
im_array_dask = imObject.to_dask()

start_t = 75
stop_t = 175
well_int = 2

data_tzyx = im_array_dask[start_t:stop_t, well_int, :, :, :].compute()

# time_points = range(15)
# im_list = []
# for t in tqdm(time_points):
#     imData = np.squeeze(imObject.get_image_data("ZYX", C=0, T=t))
#     im_list.append(imData)
#
# # Extract pixel sizes and bit_depth
res_raw = imObject.voxel_size()
scale_vec = np.asarray(res_raw)[::-1]
# im_array = np.asarray(im_list)
# shape_curr = imData.shape
# rs_factor = scale_vec[0] / scale_vec[1]
# shape_new = np.asarray(shape_curr)
# shape_new[0] = np.round(shape_new[0]*rs_factor).astype(int)
# # shape_new = np.round(shape_new / 2).astype(int) # downsize by a factor of 2
# imData_rs = resize(imData, shape_new, order=1, anti_aliasing=False)
#
# # with open("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", 'wb') as f:
# # np.save("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", imData)
#
viewer = napari.view_image(data_tzyx, colormap="magenta", scale=tuple(scale_vec))#, scale=res_array)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()