from aicsimageio import AICSImage
import numpy as np
import napari
from skimage.transform import resize
import glob2 as glob
import os
# read the image data
from ome_zarr.io import parse_url


# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/raw_data/HCR/"
# experiment_date = "20240112"
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/raw_data"
# experiment_date = "20240223"
#
# file_list = sorted(glob.glob(os.path.join(root, experiment_date, "") + "*.nd2"))
# full_filename = "/Volumes/My Passport/pec_fin_dynamics/20240223_wt_tests/B03_test_48hpf001.nd2"
full_filename = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/raw_data/20240306_tbx5aSG/tbx5a_sg_G02_20x.nd2"
print(full_filename)
imObject = AICSImage(full_filename)
imObject.set_scene(0)
imData = imObject.get_image_data("ZYX", T=0)
# imData = np.squeeze(imObject.data)

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = np.asarray(res_raw)
print(scale_vec)

viewer = napari.view_image(imData,  scale=tuple(scale_vec))#, scale=res_array)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()