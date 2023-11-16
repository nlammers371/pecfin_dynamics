from aicsimageio import AICSImage
import numpy as np
import napari
import os
from glob2 import glob
import skimage.io as skio
from alphashape import alphashape
from skimage.transform import resize
# read the image data
from ome_zarr.io import parse_url

# full_filename = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\pec_fin_dynamics\\fin_morphodynamics\\raw_data\\20230830\\tdTom_54hpf_pecfin_40x.nd2"
root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/"
date_folder = "20230830"
filename = "tdTom_54hpf_pecfin_40x"

# load metadata
ref_image_path = os.path.join(root, "raw_data", date_folder, filename + ".nd2")
imObject = AICSImage(ref_image_path)

# git list of prob files produced by CellPose
prob_file_dir = os.path.join(root, "built_data", date_folder, "")
prob_file_list = glob(prob_file_dir + filename + "*probs*")

# Extract pixel sizes and bit_depth
res_raw = imObject.physical_pixel_sizes
scale_vec = tuple(res_raw)

image_i = 0
prob_thresh = -4

def calculate_fin_hull(event):
    # print(f"{event.source.name} changed its data!")
    point_data = event.source.data
    if point_data.shape[0] > 6:
        hull = alphashape(point_data / np.max(point_data), alpha=1)
        values = np.linspace(0.5, 1, len(hull.vertices))
        surf = (hull.vertices*np.max(point_data), hull.faces, values)
        fin_hull_layer = viewer.add_surface(surf, name="fin hull", colormap="gray")
        # viewer.add_points(np.asarray([3, 100, 100]), name="test", size=15)

while image_i < len(prob_file_list):
    prob_name = prob_file_list[image_i]
    im_prob = skio.imread(prob_name, plugin="tifffile")

    viewer = napari.view_image(im_prob, colormap="magenta", scale=scale_vec, contrast_limits=(prob_thresh, np.percentile(im_prob, 99.5)))
    points_layer = viewer.add_points(np.empty((0, 3)), name="fin hull points", size=8, scale=scale_vec, n_dimensional=True)

    points_layer.events.data.connect(calculate_fin_hull)

    napari.run()

    points_layer = viewer.layers["fin hull points"]

    # print("blah")
    # wait = input(
    #     "Press Enter to continue to next image. \nPress 'x' then Enter to exit. \nPress 'n' then Enter to move to next experiment.")
    # if wait == 'x':
    #     break
    # elif wait == 'n':
    #     break
    # else:
    #     image_i += 1
    #     print(image_i)
# # labels_layer = viewer.add_labels(lbData, name='segmentation', scale=res_array)
# if __name__ == '__main__':
#     napari.run()