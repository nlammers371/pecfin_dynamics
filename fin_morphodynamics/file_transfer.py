from aicsimageio import AICSImage
import numpy as np
from tqdm import tqdm
import os
from tifffile import TiffWriter

in_path = "G:\\20231214\\tdTom_40X_pecfin_timeseries.nd2"
imObject = AICSImage(in_path)

save_directory = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\raw_data\\20231214\\backup_tiff_stacks\\"
experiment_date = "20231214"
n_wells = len(imObject.scenes)
well_list = imObject.scenes
n_time_points = imObject.dims["T"][0]

for well_index in tqdm(range(51, n_wells)):
    well_id_string = well_list[well_index]
    well_num = int(well_id_string.replace("XYPos:", ""))
    try:
        imObject.set_scene(well_id_string)
    except:
        print(f"Error setting scene for {well_num:02}")

    for t in range(n_time_points):
        # extract image
        try:
            data_zyx_raw = np.squeeze(imObject.get_image_data("CZYX", T=t))

            out_name = experiment_date + f"_well{well_num:03}_t{t:03}_raw_stack"
            out_path = os.path.join(save_directory, out_name)
            with TiffWriter(out_path + '.tif', bigtiff=True) as tif:
                tif.write(data_zyx_raw)
        except:
            print(f"Error reading or writing time {t:02} of well {well_num:02}")


# imObject.set_scene('XYPos:50')
# # for t in range(n_time_points):
# # # extract image
# t=16
# data_zyx_raw = np.squeeze(imObject.get_image_data("CZYX", T=t))