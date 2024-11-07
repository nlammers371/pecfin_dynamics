import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.utilities.data_utilities import PointDataReg
from torch.utils.data.sampler import SubsetRandomSampler
from src._Archive.point_net_nl.point_net import PointNetRegHead
from src._Archive.point_net_nl.point_net_loss import PointNetRegLoss
import torch.optim as optim
import time


def train_point_net_reg(root, training_dates, fluo_channel, num_epochs=100, learning_rate=0.0001,
                        point_cloud_size=4096, batch_size=32, train_frac=0.8):


    data_root = os.path.join(root, "built_data", "nucleus_data", "point_clouds")

    timestamp = int(time.time())
    save_path = os.path.join(root, "built_data", "nucleus_data", "point_models", str(timestamp))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # generate dataloader to load fin point clouds
    point_data = PointDataReg(data_root, split='train', training_dates=training_dates,
                                                                    npoints=point_cloud_size, fluo_channel=fluo_channel)
    
    # randomly select train and valid indices
    # np.random.seed(366)
    n_sets_total = len(point_data)
    
    n_train = np.ceil(train_frac*n_sets_total).astype(int)
    
    train_indices = np.random.choice(range(n_sets_total), n_train, replace=False)
    valid_indices = np.setdiff1d(range(n_sets_total), train_indices)
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    
    train_loader = DataLoader(point_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    valid_loader = DataLoader(point_data, batch_size=batch_size, shuffle=False, sampler=valid_sampler)


    #####################
    # prepare for training 
    
    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load pretrained model
    model = PointNetRegHead(num_points=point_cloud_size).to(device)
    # model.load_state_dict(torch.load(model_path)) # load weights from previous training as starting point
    model.eval()

    # set loss function
    model_loss = PointNetRegLoss(reduction='sum').to(device)
    
    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # vectors to store training info
    train_loss = []
    valid_loss = []
    best_loss = np.inf

    for epoch in tqdm(range(num_epochs), "Training tbx5a prediction model..."):
        # apply to FOVs to generate training features
        _train_loss = []
        for batch_i, batch in enumerate(tqdm(train_loader, "Training data...")):
            # extract data
            points = batch["data"]
            levels = batch["label"]
            points = torch.transpose(points, 1, 2)
            # levels = torch.transpose(levels, 1, 2)
    
            # pass to device
            points = points.to(device)
            levels = levels.to(device)

            # zero gradients
            optimizer.zero_grad()

            # get softmax predictions
            preds, _, _, _ = model(points)
    
            # calculate loss
            loss = model_loss(preds.float(), levels.float())

            # get loss and perform backprop
            loss.backward()
            optimizer.step()

            # update epoch loss and accuracy
            _train_loss.append(loss.item())

        train_loss.append(np.mean(_train_loss))

        print(f'Epoch: {epoch} - Train Loss: {np.mean(train_loss[-5:]):.4f}')

        with torch.no_grad():
            # apply to FOVs to generate training features
            _valid_loss = []
            for batch_i, batch in enumerate(tqdm(valid_loader, "valid data...")):
                # extract data
                points = batch["data"]
                levels = batch["label"]
                points = torch.transpose(points, 1, 2)
                # levels = torch.transpose(levels, 1, 2)

                # pass to device
                points = points.to(device)
                levels = levels.to(device)

                # get softmax predictions
                preds, _, _, _ = model(points)

                # calculate loss
                loss = model_loss(preds, levels)

                # update epoch loss and accuracy
                _valid_loss.append(loss.item())

            valid_loss.append(np.mean(_valid_loss))

            print(f'Epoch: {epoch} - valid Loss: {np.mean(valid_loss[-5:]):.4f}')

            # save best models
            if np.mean(valid_loss[-5:]) < best_loss:
                best_loss = np.mean(valid_loss[-5:])
                torch.save(model.state_dict(), os.path.join(save_path, f'seg_model_{epoch:04}.pth'))

    # save training loss info
    loss_df = pd.DataFrame(np.arange(num_epochs), columns=["epoch"])
    loss_df['train_loss'] = train_loss
    loss_df['valid_loss'] = valid_loss
    loss_df.to_csv(os.path.join(save_path, 'train_loss.csv'))



if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    fluo_channel = "tbx5a-StayGold_mean_nn"
    experiment_dates = ["20240424", "20240425"]

    # build point cloud files
    train_point_net_reg(root, training_dates=experiment_dates, fluo_channel=fluo_channel, num_epochs=250)