from aicsimageio import AICSImage
import numpy as np
import napari
from skimage.transform import resize
# read the image data
import os
from ome_zarr.io import parse_url

# full_filename = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\pec_fin_dynamics\\fin_morphodynamics\\raw_data\\20230830\\tdTom_54hpf_pecfin_40x.nd2"
root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecFin/YX1_HCR/20231119/HCR/20231120/"
image_name = "D3_72hpf_20x_405_DAPI_546_Myod_640_Emilin.nd2"
full_filename = os.path.join(root, image_name)
imObject = AICSImage(full_filename)
# imObject.set_scene("XYPos:6")
imData = np.squeeze(imObject.get_image_data("CZYX"))
wvl_vec = imObject.channel_names

wvl_vec_full = ["405", "561", "640"]
channel_names_full = ["DAPI", "Myod", "Emilin"]
colors_full = ["gray", "green", "magenta"]

channel_names = [channel_names_full[c] for c in range(len(channel_names_full)) if wvl_vec_full[c] in wvl_vec]
color_names = [colors_full[c] for c in range(len(colors_full)) if wvl_vec_full[c] in wvl_vec]

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = np.asarray(res_raw)

shape_curr = imData.shape

#
# # with open("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", 'wb') as f:
# # np.save("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", imData)
#
viewer = napari.view_image(imData, channel_axis=0, name=channel_names, colormap=color_names, scale=scale_vec)#, scale=res_array)

# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()