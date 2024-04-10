from tqdm import tqdm
import numpy as np
import napari
import glob2 as glob
import os
import skimage.io as io
import nd2

# set paths
# nd2_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\raw_data\\20231214\\tdTom_40X_pecfin_timeseries.nd2"
nd2_path = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
# zarr_path = f"E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\built_data\zarr_image_files\\20240223\\20240223_well0002.zarr"
well_int = 2

# extract key metadata info
imObject = nd2.ND2File(nd2_path)
im_array_dask = imObject.to_dask()
nd2_shape = im_array_dask.shape
n_time_points = nd2_shape[0]
n_wells = nd2_shape[1]
dtype = im_array_dask.dtype
res_raw = imObject.voxel_size()
scale_vec = np.asarray(res_raw)[::-1]

time_range = np.asarray(np.arange(180).tolist() + np.arange(181, 202).tolist())
# load zyx stack

data_tzyx = np.squeeze(im_array_dask[time_range, well_int, :, :, :].compute())

# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(data_tzyx, scale=tuple(scale_vec))
# viewer.add_image(z_depth_stack, scale=scale_vec_plot, opacity=0.5, colormap="magma")
viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(label_stack, scale=scale_vec_plot)
# movie = Movie(myviewer=viewer)
napari.run()


# if __name__ == '__main__':
#     napari.run()