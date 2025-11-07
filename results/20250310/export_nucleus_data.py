# import os
# import sys
#
# code_root = "/net/trapnell/vol1/home/nlammers/projects/repositories/pecfin_dynamics"
# sys.path.insert(0, code_root)

from src.nucleus_dynamics.export_to_zarr.export_nd2_to_zarr import export_nd2_to_zarr

experiment_date_vec = ["20250225"]
# root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
# root = "/net/trapnell/vol1/home/nlammers/projects/data/pecfin_dynamics/"
root = "Y:\\projects\\data\\pecfin_dynamics\\"
overwrite_flag = True
nuclear_channel_vec = [0]  #, 1, 1]
channel_names_vec = [["H2B-tdTom"]]  #, ["tbx5a-StayGold", "H2B-tdTom"]]

for e, experiment_date in enumerate(experiment_date_vec):

    print("#########################")
    print("Exporting nucleus data for experiment {}".format(experiment_date))
    print("#########################")

    nuclear_channel = nuclear_channel_vec[e]
    channel_names = channel_names_vec[e]
    export_nd2_to_zarr(root, experiment_date, overwrite_flag, nuclear_channel=nuclear_channel, channel_names=channel_names)