from tqdm import tqdm
import numpy as np
import napari
import glob2 as glob
import os
import skimage.io as io
import nd2
from skimage.morphology import label
from skimage.measure import regionprops
import zarr
from scipy.spatial import KDTree
import skimage.graph as graph
from skimage.transform import resize
import skimage as ski
import SimpleITK as sitk
import networkx as nx
import pandas as pd

def process_raw_image(data_zyx, scale_vec):
    # estimate background using blur
    top1 = np.percentile(data_zyx, 99)
    data_capped = data_zyx.copy()
    data_capped[data_capped > top1] = top1
    # data_capped = data_capped[:, 500:775, 130:475]

    shape_orig = np.asarray(data_capped.shape)
    shape_iso = shape_orig.copy()
    iso_factor = scale_vec[0] / scale_vec[1]
    shape_iso[0] = shape_iso[0] * iso_factor

    gaussian_background = ski.filters.gaussian(data_capped, sigma=(2, 8, 8))
    data_1 = np.divide(data_capped, gaussian_background)
    # data_sobel = ski.filters.sobel(data_1)
    # data_sobel_i = ski.util.invert(data_sobel)

    data_rs = resize(data_1, shape_iso, preserve_range=True, order=1)
    image = sitk.GetImageFromArray(data_rs)
    data_log = sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(image, sigma=1))
    data_log_i = resize(ski.util.invert(data_log), shape_orig, preserve_range=True, order=1)
    # data_log_i = ski.util.invert(data_log)

    # rescale and convert to 16 bit
    data_bkg_16 = data_1.copy()
    data_bkg_16 = data_bkg_16 - np.min(data_bkg_16)
    data_bkg_16 = np.round(data_bkg_16 / np.max(data_bkg_16) * 2 ** 16 - 1).astype(np.uint16)

    log_i_16 = data_log_i.copy()
    log_i_16 = log_i_16 - np.min(log_i_16)
    log_i_16 = np.round(log_i_16 / np.max(log_i_16) * 2 ** 16 - 1).astype(np.uint16)

    return log_i_16, data_bkg_16

def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


# set paths
root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"

experiment_date = "20240223"
model2 = "log-v5"
label_directory2 = os.path.join(root, "built_data", "cellpose_output", model2, experiment_date, '')
data_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')


# get list of images
image_list = sorted(glob.glob(data_directory + "*.zarr"))

# zarr_path = "F://pec_fin_dynamics//20240223_wt_tests//wt_tdTom_timelapse_long.nd2"
time_int = 201
well_int = 2

# load labels and probs
prob_name = experiment_date + f"_well{well_int:03}_t{time_int:03}_probs.tif"
# prob_path1 = os.path.join(label_directory1, prob_name)
# im_prob1 = io.imread(prob_path1)
label_name = experiment_date + f"_well{well_int:03}_t{time_int:03}_labels.tif"
# label_path1 = os.path.join(label_directory1, label_name)
# im_label1 = io.imread(label_path1)

prob_path2 = os.path.join(label_directory2, prob_name)
im_prob2 = io.imread(prob_path2)
label_path2 = os.path.join(label_directory2, label_name)
im_label2 = io.imread(label_path2)

# generate DF with mask labels
label_index = np.unique(im_label2)
well_id_vec = np.ones(label_index.shape)*well_int
time_id_vec = np.ones(label_index.shape)*time_int

label_df = pd.DataFrame(well_id_vec, columns=["well_id"])
label_df["time_id"] = time_id_vec
label_df["experiment_date"] = experiment_date
label_df["nucleus_label"] = label_index
label_df["quality_score"] = -1

label_directory = os.path.join(root, "built_data", "nucleus_curation", model2, "raw_lb_files", '')
if not os.path.isdir(label_directory):
    os.makedirs(label_directory)
label_df.to_csv(os.path.join(label_directory, experiment_date + "_label_df.csv"), index=False)


# extract key metadata info
data_tzyx = zarr.open(image_list[well_int], mode="r")
data_zyx = data_tzyx[time_int]

dtype = data_zyx.dtype
scale_vec = tuple([2.0, 0.55, 0.55])


# generate napari viewer instance
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(data_zyx, scale=scale_vec)
# viewer.add_labels(im_label1, scale=scale_vec, name="bkg")
viewer.add_labels(im_label2, scale=scale_vec, name="log")
# viewer.add_labels(label(im_prob > -4), scale=scale_vec)


log_data, bkg_data = process_raw_image(data_zyx, scale_vec)

# get mask graph stats
regions = regionprops(im_label2)
c_array = np.asarray([rg["Centroid"] for rg in regions])
mask_tree = KDTree(c_array)
dd, ii = mask_tree.query(c_array, k=10)

edges = -im_prob2.copy()
edges[im_label2 == 0] = 1e5
g = graph.rag_boundary(im_label2, edges, connectivity=2)

# remove background
g.remove_node(0)
g2 = g.copy()

# add region areas
area_dict = dict({})
for r in range(1, len(regions)):
    rg = regions[r]
    area_dict[r] = rg["Area"]

nx.set_node_attributes(g2, area_dict, "Volume")

edge_list = list(g2.edges())
ba_vec = np.empty((len(edge_list,)))

for i, e in enumerate(edge_list):
    ba = g2[e[0]][e[1]]["count"]
    mv = np.min([regions[e[0]-1]["Area"], regions[e[1]-1]["Area"]])
    ba_norm = ba / mv
    ba_vec[i] = ba_norm
    g2[e[0]][e[1]]["weight"] = -ba_norm

# im_label2_v2 = graph.merge_hierarchical(im_label2, g2, thresh=-6, rag_copy=False,
#                                    in_place_merge=True,
#                                    merge_func=merge_boundary,
#                                    weight_func=weight_boundary)

# im_label2_v2 = graph.cut_threshold(im_label2, g2, thresh=-0.29, in_place=True)
#
# viewer.add_labels(im_label2_v2, scale=scale_vec, name="log-stitched")
# viewer.add_labels(label(im_prob>=2), scale=scale_vec, name="prob thresh labels") #, contrast_limits=[-16, 16])
# viewer.add_image(z_depth_stack, scale=scale_vec_plot, opacity=0.5, colormap="magma")
# viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
# viewer.add_labels(label_stack, scale=scale_vec_plot)
# movie = Movie(myviewer=viewer)
napari.run()


# if __name__ == '__main__':
#     napari.run()