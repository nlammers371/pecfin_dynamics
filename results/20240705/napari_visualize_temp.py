from tqdm import tqdm
import numpy as np
import napari
import zarr



# set paths
prob_path = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/20240620_well0000_probs.zarr"
prob_zarr = zarr.open(prob_path, mode="r")
mask_path0 = "E:/Nick\Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/mask_stacks/tdTom-bright-log-v5/20240620_rs05/20240620_well0000_mask_stacks.zarr"
mask_zarr0 = zarr.open(mask_path0, mode="r")
mask_path1 = "E:/Nick\Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/mask_stacks/tdTom-bright-log-v5/20240620/20240620_well0000_mask_stacks.zarr"
mask_zarr1 = zarr.open(mask_path1, mode="r")
time_ind = 0
thresh_ind = 0
scale_vec = tuple(mask_zarr0.attrs["voxel_size_um"])


# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(prob_zarr[time_ind], scale=scale_vec)
# viewer.add_labels(mask_s_zarr[time_ind], scale=scale_vec)
viewer.add_labels(mask_zarr0[time_ind], scale=scale_vec, name="full-res")
viewer.add_labels(mask_zarr1[time_ind], scale=scale_vec, name="rs05")
# viewer.add_labels(im_label1, scale=scale_vec, name="bkg")
# viewer.add_labels(mask_zarr[time_ind], scale=scale_vec)
# viewer.add_labels(stitched_mask, name="aff labels")
# viewer.add_labels(stitched_mask_p6, name="aff labels p6")
# viewer.add_labels(stitched_mask_n6, name="aff labels n6")

# viewer.add_labels(label(im_prob > -4), scale=scale_vec)


# if __name__ == '__main__':
#     napari.run()