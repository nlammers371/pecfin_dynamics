from src.tracking.perform_tracking_v2 import perform_tracking

if __name__ == '__main__':
    experiment_date = "20240620"
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    # well_num = 3
    use_centroids = False
    tracking_config = "tracking_jordao_20240918.txt"
    segmentation_model = "tdTom-bright-log-v5"
    add_label_spacer = False

    for well_num in range(14, 16):
        print("######################################")
        print(f"##############WELL{well_num:03}#################")
        print("######################################")
        perform_tracking(root, experiment_date, well_num, tracking_config,
                     seg_model=segmentation_model, start_i=0, stop_i=None, overwrite_registration=False,
                         overwrite_tracking=False)