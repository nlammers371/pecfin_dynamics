# import sys
# import site
#
# user_site = site.getusersitepackages()
# if user_site in sys.path:
#     sys.path.remove(user_site)

import numpy as np
import napari
from aicsimageio import AICSImage
from readlif.reader import LifFile
import bioio_lif
import dask.array as da

# set paths
# nd2_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\raw_data\\20231214\\tdTom_40X_pecfin_timeseries.nd2"
im_path = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/raw_data/amullen/LIFFiles/Pax7Phallodin_Mutant_24hpf_3_23_24/031624_Pax7_Phallodin.lif"
# zarr_path = f"E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\built_data\zarr_image_files\\20240223\\20240223_well0002.zarr"
well_int = 0

# extract key metadata info
imObject = LifFile(im_path)
# imObject = AICSImage(im_path)
im_array_dask = imObject.to_dask()
# Persist the array so that its data is loaded while the file is open
# im_array_dask = im_array_dask.persist()
# (Optionally, wait for persistence to finish)
# im_array = im_array_dask.compute()
# Extract metadata as needed
dim_dict = imObject.sizes
res_raw = imObject.voxel_size()
data_ctzyx = np.squeeze(im_array_dask[:, 100:110, well_int, :, :, :])

# n_time_points = dim_dict["T"]
# n_wells = dim_dict["P"]
dim_order = list(dim_dict.keys())

target_order = ['C', 'T', 'P', 'Z', 'Y', 'X']
target_order = [d for d in target_order if d in dim_order]
permute_vec = [dim_order.index(target_order[d]) for d in range(len(target_order))]
im_array_dask = da.transpose(im_array_dask, tuple(permute_vec))

dtype = im_array_dask.dtype

scale_vec = np.asarray(res_raw)[::-1]

# load zyx stack



# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(data_ctzyx, scale=tuple(scale_vec), channel_axis=0)
# viewer.add_image(z_depth_stack, scale=scale_vec_plot, opacity=0.5, colormap="magma")
viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(label_stack, scale=scale_vec_plot)
# movie = Movie(myviewer=viewer)
napari.run()



# if __name__ == '__main__':
#     napari.run()