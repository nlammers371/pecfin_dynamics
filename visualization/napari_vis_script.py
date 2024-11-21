from tqdm import tqdm
import numpy as np
import napari
import zarr



# set paths
# zarr_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/zarr_image_files/20240425/"
zarr_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/stitched_labels/log-v5/20240424/20240424_well0017_labels_stitched.zarr/"
# zarr_list = sorted(glob.glob(zarr_path + "*.zarr"))

# zarr_ind = 8

raw_zarr = zarr.open(zarr_path, mode="r")
scale_vec = tuple([2.0, 0.55, 0.55])
time = 40

# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)

# viewer.add_labels(im_label1, scale=scale_vec, name="bkg")
viewer.add_labels(raw_zarr, scale=scale_vec)
# viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(stitched_mask, name="aff labels")
# viewer.add_labels(stitched_mask_p6, name="aff labels p6")
# viewer.add_labels(stitched_mask_n6, name="aff labels n6")

# viewer.add_labels(label(im_prob > -4), scale=scale_vec)


# if __name__ == '__main__':
#     napari.run()