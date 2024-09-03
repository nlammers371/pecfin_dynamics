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

# set paths
mask_path = "Y:/data/pecfin_dynamics/built_data/zarr_image_files/20240726/20240726_well0010.zarr"
mask_zarr = zarr.open(mask_path, mode="r")
time_ind = 0
scale_vec = tuple([2.0, 0.55, 0.55])

# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)

# viewer.add_labels(im_label1, scale=scale_vec, name="bkg")
viewer.add_image(mask_zarr[:, time_ind], scale=scale_vec, channel_axis=0, colormap=["Green", "magenta"], name=["fli-GFP", "H2B-tdTom"], contrast_limits=[0, 2**16])

viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')

napari.run()
