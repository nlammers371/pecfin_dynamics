from aicsimageio import AICSImage
import numpy as np
import napari
from skimage.transform import resize
import glob2 as glob
import os
# read the image data
from ome_zarr.io import parse_url


root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/raw_data/HCR/"
experiment_date = "20240112"

file_list = sorted(glob.glob(os.path.join(root, experiment_date, "") + "*.nd2"))
full_filename = file_list[4]
print(full_filename)
imObject = AICSImage(full_filename)
# imObject.set_scene("XYPos:6")
imData = np.squeeze(imObject.data)

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = np.asarray(res_raw)
print(scale_vec)

viewer = napari.view_image(imData, channel_axis=0, scale=scale_vec)#, scale=res_array)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()