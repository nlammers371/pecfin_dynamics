# import sys
# import site
#
# user_site = site.getusersitepackages()
# if user_site in sys.path:
#     sys.path.remove(user_site)

from aicsimageio import AICSImage
from tqdm import tqdm
import zarr
import os

# read path
im_path = "/home/nick/Documents/Mullen/CycloDKOExperiments/01_30_25_Cyclo_Myod1Six1ab_DAPI_F310.lif"

# save path
project_name = "amullen"
out_dir = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/zarr_image_files/"
save_path = os.path.join(out_dir, project_name, "")

# extract key metadata info
imObject = AICSImage(im_path)
scene_list = list(imObject.scenes)
actual_images = [im for im in scene_list if ("Series" not in im) and ("Image" not in im)]
actual_indices = [i for i in range(len(scene_list)) if ("Series" not in scene_list[i]) and ("Image" not in scene_list[i])]

# load image
for a, ind in enumerate(tqdm(actual_indices, "Exporting images to zarr...")):
    # set the scene
    imObject.set_scene(ind)
    im_name = actual_images[a]
    # load pixel sizes and other metadata
    px_sizes = imObject.physical_pixel_sizes
    channels = imObject.channel_names
    # load image
    im_data = imObject.data
    # initialize zarr object
    zarr_path = os.path.join(save_path, im_name + ".zarr")
    im_zarr = zarr.open(zarr_path, mode="w", shape=im_data.shape)
    # add image
    im_zarr[:] = im_data
    # add metadata
    im_zarr.attrs["channels"] = channels
    im_zarr.attrs["voxel_size_um"] = px_sizes
# if __name__ == '__main__':
#     napari.run()