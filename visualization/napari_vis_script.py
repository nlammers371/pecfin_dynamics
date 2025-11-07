from tqdm import tqdm
import numpy as np
import napari
import zarr
import os


name = "amullen_well0000"
# set paths
root = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/cellpose_output/DAPI-Pro-7/amullen/"
label_path = os.path.join(root, name + "_labels.zarr")
prob_path = os.path.join(root, name + "_probs.zarr")
im_path = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/zarr_image_files/amullen/24hpf_Cyclopamine_DAPI_F310_Multinucleate.zarr"
# zarr_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/stitched_labels/log-v5/20240424/20240424_well0017_labels_stitched.zarr/"
# zarr_list = sorted(glob.glob(zarr_path + "*.zarr"))

# zarr_ind = 8
im_zarr = zarr.open(im_path, mode="r")
mask_zarr = zarr.open(label_path, mode="r")
prob_zarr = zarr.open(prob_path, mode="r")
scale_vec = tuple(prob_zarr.attrs['voxel_size_um'])
# time = 40

# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)

# viewer.add_labels(im_label1, scale=scale_vec, name="bkg")
viewer.add_image(im_zarr[0], channel_axis=0, name="raw_image", scale=scale_vec)
viewer.add_image(prob_zarr, name="probs", scale=scale_vec)
viewer.add_labels(mask_zarr, name="labels", scale=scale_vec)
# viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(stitched_mask, name="aff labels")
# viewer.add_labels(stitched_mask_p6, name="aff labels p6")
# viewer.add_labels(stitched_mask_n6, name="aff labels n6")
napari.run()
# viewer.add_labels(label(im_prob > -4), scale=scale_vec)


# if __name__ == '__main__':
#     napari.run()