''' Point Net Loss function which is essentially a regularized Focal Loss.
    Code was adapted from this repo:
        https://github.com/clcarwin/focal_loss_pytorch
    '''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




# special loss for segmentation Focal Loss + Dice Loss
class PointNetRegLoss(nn.Module):
    def __init__(self, reduction="sum"):
        super(PointNetRegLoss, self).__init__()

        # self.size_average = size_average

        # get Balanced Cross Entropy Loss
        self.mse_loss = nn.MSELoss(reduction=reduction)
        

    def forward(self, predictions, targets):

        # get Balanced Cross Entropy Loss
        loss = self.mse_loss(predictions, targets)

        return loss


    @staticmethod
    def dice_loss(predictions, targets, eps=1):
        ''' Compute Dice loss, directly compare predictions with truth '''

        targets = targets.reshape(-1)
        predictions = predictions.reshape(-1)

        cats = torch.unique(targets)

        top = 0
        bot = 0
        for c in cats:
            locs = targets == c

            # get truth and predictions for each class
            y_tru = targets[locs]
            y_hat = predictions[locs]

            top += torch.sum(y_hat == y_tru)
            bot += len(y_tru) + len(y_hat)


        return 1 - 2*((top + eps)/(bot + eps)) 