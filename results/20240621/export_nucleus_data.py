from src.build.build00_export_nd2_to_zarr import export_nd2_to_zarr

experiment_date_vec = ["20240425"] #, "20240424", "20240425"]
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
overwrite_flag = False
nuclear_channel_vec = [1=]  #, 1, 1]
channel_names_vec = [["tbx5a-StayGold", "H2B-tdTom"]]#, ["tbx5a-StayGold", "H2B-tdTom"]]
for e, experiment_date in enumerate(experiment_date_vec):

    print("#########################")
    print("Exporting nucleus data for experiment {}".format(experiment_date))
    print("#########################")

    nuclear_channel = nuclear_channel_vec[e]
    channel_names = channel_names_vec[e]
    export_nd2_to_zarr(root, experiment_date, overwrite_flag, nuclear_channel=nuclear_channel, channel_names=channel_names)