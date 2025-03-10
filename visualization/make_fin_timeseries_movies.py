# import sys
# import site
#
# user_site = site.getusersitepackages()
# if user_site in sys.path:
#     sys.path.remove(user_site)

import numpy as np
import napari
import zarr
import nd2
import dask.array as da
from tqdm import tqdm
from skimage import exposure

# set paths
# nd2_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\raw_data\\20231214\\tdTom_40X_pecfin_timeseries.nd2"
# nd2_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/raw_data/20250122/SG-tbx5a_tdTom-H2B_timelapse.nd2"
# nd2_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\raw_data\\20250122\\SG-tbx5a_tdTom-H2B_timelapse.nd2"
zarr_path = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\built_data\\zarr_image_files_registered\\20240424\\20240424_well0012.zarr"
# zarr_path = f"E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\built_data\zarr_image_files\\20240223\\20240223_well0002.zarr"
# well_int = 1

# extract key metadata info
# imObject = nd2.ND2File(nd2_path)
# im_array_dask = imObject.to_dask()
# Persist the array so that its data is loaded while the file is open
# im_array_dask = im_array_dask.persist()
# (Optionally, wait for persistence to finish)
# im_array = im_array_dask.compute()
# Extract metadata as needed
# dim_dict = imObject.sizes
# res_raw = imObject.voxel_size()

im_zarr = zarr.open(zarr_path, mode="r")

# n_time_points = dim_dict["T"]
# n_wells = dim_dict["P"]
# dim_order = list(dim_dict.keys())
#
# target_order = ['C', 'T', 'P', 'Z', 'Y', 'X']
# target_order = [d for d in target_order if d in dim_order]
# permute_vec = [dim_order.index(target_order[d]) for d in range(len(target_order))]
# im_array_dask = da.transpose(im_array_dask, tuple(permute_vec))
#
# data_ctzyx = np.squeeze(im_array_dask[:, :, well_int, :, :, :]).compute()
#
# dtype = im_array_dask.dtype
#
# scale_vec = np.asarray(res_raw)[::-1]
scale_vec = tuple(im_zarr.attrs['voxel_size_um'])

# Choose a reference histogram – e.g. the first frame:
reference = np.asarray(im_zarr[1, 0])

# Create an output array for the normalized frames
# normalized = np.empty_like(timeseries)

# Match each frame's histogram to the reference histogram:
im_zarr_norm = np.asarray(im_zarr).copy()
for t in tqdm(range(im_zarr.shape[1]), "Normalizing H2B channel..."):
    im_zarr_norm[1, t] = exposure.match_histograms(im_zarr[1, t], reference)

# normalize the H2B signal to offset bleaching
# for t in tqdm(range(im_zarr.shape[1]), "Normalizing nuclear channel..."):
#     nuc_frame = im_zarr_norm[1, t].copy().astype(float)
#     nuc_frame = (nuc_frame / np.percentile(nuc_frame, 99.9)) * 2**16
#     im_zarr_norm[1, t] = nuc_frame.astype(np.int16)

# calculate contrast scales
c0 = (0, np.max(im_zarr_norm[0, -1]))
c1 = (0, np.percentile(im_zarr_norm[1, 0], 99.9))

# generate napari viewer instancescale_)
viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(data_ctzyx, scale=tuple(scale_vec), channel_axis=0)
viewer.add_image(im_zarr_norm, scale=tuple(scale_vec), channel_axis=0, contrast_limits=[c0, c1])
# viewer.add_image(z_depth_stack, scale=scale_vec_plot, opacity=0.5, colormap="magma")
viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(label_stack, scale=scale_vec_plot)
# movie = Movie(myviewer=viewer)
napari.run()



# if __name__ == '__main__':
#     napari.run()