from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import napari
import os
import numpy as np
from aicsimageio import AICSImage
from skimage.transform import resize
import tifffile

# set parameters
filename = "2022_12_21 HCR Prdm1a Robo3 Fgf10a_1"  #"2022_12_15 HCR Hand2 Tbx5a Fgf10a_3"
db_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\pecFin\\HCR_Data\\"
# db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/"
readPath = os.path.join(db_path, 'raw', filename[:-2], filename + "_decon.czi")

#############
# Main image
#############

# read the image data
# reader = Reader(parse_url(readPathMeta))

# import tifffile

imObject = AICSImage(readPath)
image_data = np.squeeze(imObject.data)
scale_vec = np.asarray(imObject.physical_pixel_sizes)

viewer = napari.view_image(image_data, channel_axis=0, colormap=["gray", "green", "blue", "red"][::-1], scale=tuple(scale_vec))
#
viewer.theme = "dark"



if __name__ == '__main__':
    napari.run()