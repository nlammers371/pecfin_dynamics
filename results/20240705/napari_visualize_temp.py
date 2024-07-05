from tqdm import tqdm
import numpy as np
import napari
import zarr



# set paths
prob_path = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/20240620_well0000_probs.zarr"
prob_zarr = zarr.open(prob_path, mode="r")
time_ind = 2
scale_vec = tuple([2.0, 0.55, 0.55])


# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(prob_zarr[-1], scale=scale_vec)
# viewer.add_labels(im_label1, scale=scale_vec, name="bkg")
# viewer.add_labels(mask_zarr[time_ind], scale=scale_vec)
# viewer.add_labels(stitched_mask, name="aff labels")
# viewer.add_labels(stitched_mask_p6, name="aff labels p6")
# viewer.add_labels(stitched_mask_n6, name="aff labels n6")

# viewer.add_labels(label(im_prob > -4), scale=scale_vec)


# if __name__ == '__main__':
#     napari.run()