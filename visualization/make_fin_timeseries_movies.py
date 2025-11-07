# import sys
# import site
#
# user_site = site.getusersitepackages()
# if user_site in sys.path:
#     sys.path.remove(user_site)

import numpy as np
import napari
from aicsimageio import AICSImage
# from bioio import BioImage
import bioio_lif
from readlif.reader import LifFile
import dask.array as da

# set paths

im_path = "/home/nick/Documents/Mullen/CycloDKOExperiments/01_30_25_Cyclo_Myod1Six1ab_DAPI_F310.lif"


# extract key metadata info
# imObject = LifFile(im_path)
imObject = AICSImage(im_path)
scene_list = list(imObject.scenes)
actual_images = [im for im in scene_list if ("Series" not in im) and ("Image" not in im)]
actual_indices = [i for i in range(len(scene_list)) if ("Series" not in scene_list[i]) and ("Image" not in scene_list[i])]

# load image
ind = actual_indices[0]
imObject.set_scene(ind)

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