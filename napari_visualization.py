from aicsimageio import AICSImage
import numpy as np
import napari
import zarr


# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/raw_data/HCR/"
# experiment_date = "20240112"
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/raw_data"
# experiment_date = "20240223"
#
# file_list = sorted(glob.glob(os.path.join(root, experiment_date, "") + "*.nd2"))
# full_filename = "/Volumes/My Passport/pec_fin_dynamics/20240223_wt_tests/B03_test_48hpf001.nd2"
full_filename = "Y:\\data\\pecfin_dynamics\\built_data\\cellpose_output\\tdTom-dim-log-v3\\20240425\\20240425_well0001_probs.zarr"
# full_filename = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/zarr_image_files/20240619/20240619_well0000.zarr"
prob_zarr = zarr.open(full_filename, mode="r")
# imData = np.squeeze(imObject.data)

# Extract pixel sizes and bit_depth
# res_raw = prob_zarr.attrs["voxel_size_um"]

viewer = napari.view_image(prob_zarr[10],  scale=tuple([1.5, 0.55, 0.55]))
# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
if __name__ == '__main__':
    napari.run()