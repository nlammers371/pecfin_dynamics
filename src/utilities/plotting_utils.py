import plotly
import numpy as np


def setup_environment(fig_path):
    import os
    import plotly
    import plotly.graph_objects as go
    from IPython.display import display, HTML
    import numpy as np
    import os
    import pandas as pd
    from src.utilities.fin_shape_utils import plot_mesh
    from src.utilities.fin_class_def import FinData
    from src.utilities.functions import path_leaf
    import glob2 as glob
    import trimesh
    from tqdm import tqdm
    import torch
    import geomloss

    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plotly.offline.init_notebook_mode()
    display(HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    ))


def get_chemi_colors():
    # paired_colors = np.asarray([
    #     [253, 191, 111],  # light orange
    #     [255, 127, 0],  # orange
    #     [200, 200, 200],  # light gray
    #     [133, 133, 133],  # gray
    #     [66, 66, 66],  # dark gray
    #     [166, 206, 227],  # light blue
    #     [31, 120, 180],  # blue
    #     [178, 223, 138],  # light green
    #     [51, 160, 44],  # green
    #     [251, 154, 153],  # light red
    #     [239, 90, 91],  # med red
    #     [227, 26, 28],  # red
    #     [202, 178, 214],  # light purple
    #     [106, 61, 154],  # purple
    #     # [55, 255, 153],  # light yellow
    #     # [177, 89, 40]  # brown
    # ]) / 255

    # paired_color_list = [f"rgb({p[0]}, {p[1]}, {p[2]})" for p in paired_colors.tolist()]
    # paired_color_list += ["olive", "darkolivegreen", "olivedrab"]
    # paired_color_list += ["lightcoral", "coral"]

    dark2 = plotly.colors.qualitative.Dark2
    set2 = plotly.colors.qualitative.Set2
    pastel2 = plotly.colors.qualitative.Pastel2

    color_dict = dict({
        "DMSO_24": "lightgray",
        "DMSO_30": "gray",
        "DMSO_36": "darkgray",
        "DMSO_42": "rgb(60, 60, 60)",
        "Bmp_24": "rgb(253, 205, 172)",
        "Bmp_30": "rgb(252, 141, 98)",
        "Bmp_36": "rgb(217, 95, 2)",
        "Bmp_42": "rgb(162, 71, 1)",  # Fix
        "Wnt_24": "rgb(179, 266, 205)",
        "Wnt_30": "rgb(102, 194, 165)",
        "Wnt_36": "rgb(27, 158, 119)",
        "Wnt_42": "rgb(20, 118, 89)",  # Fix
        "fgf_24": pastel2[-2],
        "fgf_30": set2[-2],
        "fgf_36": dark2[-2],
        "fgf_42": 'rgb(124,88,21)',
        "notch_24": pastel2[-3],
        "notch_30": set2[-3],
        "notch_36": dark2[-3],
        "notch_42": 'rgb(172,128,1)',
        "ra_24": pastel2[-4],
        "ra_30": set2[-4],
        "ra_36": dark2[-4],
        "ra_42": 'rgb(76,124,22)',
        "shh_24": pastel2[-5],
        "shh_30": set2[-5],
        "shh_36": dark2[-5],
        "shh_42": 'rgb(173,30,103)',
        "tgfb_24": pastel2[-6],
        "tgfb_30": set2[-6],
        "tgfb_36": dark2[-6],
        "tgfb_42": 'rgb(87, 84, 134)'
    }
    )

    return color_dict