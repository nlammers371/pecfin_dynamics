import numpy as np
import os
import glob2 as glob
from skimage import io
from tifffile import TiffWriter, TiffReader
from tqdm import tqdm

if __name__ == "__main__":
    # sert some hyperparameters
    overwrite = False
    model_type = "nuclei"
    output_label_name = "td-Tomato"
    seg_channel_label = "561"
    xy_ds_factor = 1

    # set path to CellPose model to use
    pretrained_model = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\built_data\\cellpose_output\\20231013\\CellPoseModel\\tdTom_20231013_v3"

    # set read/write paths
    root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
    experiment_date = "20231214"
    # if tiff

    tiff_directory = os.path.join(root, "built_data", "cellpose_output", experiment_date, '')

    image_list = sorted(glob.glob(tiff_directory + "*_labels.tif"))

    for i, image_path in enumerate(tqdm(image_list)):

        im = io.imread(image_path)
        im_arr = np.asarray(im)

        im_arr = im_arr.astype(np.uint16)

        # with TiffWriter(image_path, bigtiff=True) as tif:
        #     tif.write(im_arr)

        # tif.close()
        try:
            io.imsave(image_path, im_arr)
        except:
            pass

        del im, im_arr