import napari
import nd2  # from the `nd2` package
import dask.array as da
import numpy as np
from pathlib import Path


path = Path("Y:\\projects\\data\\killi_tracker\\raw_data\\YX1\\20250621\\kf_nls_bc1_10x_series.nd2")

"""Open ND2 lazily (T, C, Z, Y, X) → (Z, Y, X) or (C, Z, Y, X) dask array."""
nd = nd2.ND2File(path)  # context manager not strictly needed; napari holds ref
shape = nd.shape        # e.g. (T, H, W, Z, C) or (T, W, Z, C, Y, X)

# convert to a dask array on‐the‐fly
# nd2 puts axes in this order: T, Y, X, Z, C  *if all are present*
# You can pick the subset you want. Here we keep everything.
darr = nd.to_dask()

          # crude check for “likely channels”
darr = da.moveaxis(darr, -3, 2)  # C axis right after T
meta = {
    "scale": nd.voxel_size()[::-1],  # (Z, Y, X) µm per pixel
    "name": path.name
}
# Extract pixel sizes and bit_depth
# res_raw = prob_zarr.attrs["voxel_size_um"]
sample_id = 1
sample_well = darr[:, sample_id, :, :, :].compute()  # e.g. (T, Z, Y, X) or (C, T, Z, Y, X)
viewer = napari.view_image(sample_well, channel_axis=1, scale=tuple([3.5, 0.55, 0.55]), contrast_limits=[0, 2500])
# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()
    print("check")