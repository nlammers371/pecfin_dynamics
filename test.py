import sys
sys.path.append('/home/nick/projects')
# from PointGPT import segmentation
from SMF_public.loader import FinDataset
import numpy as np
import pandas as pd
import os
from glob2 import glob
import plotly.express as px
import plotly.graph_objects as go
import torch


root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/point_cloud_data/vae_training_20240825/"
dataset = FinDataset(root=root, split="train")

loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 32,
            num_workers = 2,
            shuffle = True
            )

next(iter(loader))
