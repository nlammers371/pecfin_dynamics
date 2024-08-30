from tqdm import tqdm
import numpy as np
import napari
import glob2 as glob
import os
import skimage.io as io
import nd2
from skimage.morphology import label
from skimage.measure import regionprops
import zarr
import dask.array as da


# set paths
mask_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/hcr/peripheral_nerves_bmpi/DAPI_GFP_bmpi_fin_series.nd2"

im_object = nd2.ND2File(mask_path)
im_array_dask = im_object.to_dask()
im_array_dask = da.transpose(im_array_dask, (0, 2, 1, 3, 4))

time_ind = 2
scale_vec = tuple([2.0, 0.55, 0.55])

# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)

# viewer.add_labels(im_label1, scale=scale_vec, name="bkg")
viewer.add_image(np.squeeze(im_array_dask[0]), scale=scale_vec, channel_axis=0, colormap=["magenta", "Green"])
# viewer.add_labels(stitched_mask, name="aff labels")
# viewer.add_labels(stitched_mask_p6, name="aff labels p6")
# viewer.add_labels(stitched_mask_n6, name="aff labels n6")

# viewer.add_labels(label(im_prob > -4), scale=scale_vec)


if __name__ == '__main__':
    napari.run()