import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from losses.perceptual import PerceptualLoss
from losses.reconstruction import ReconstructionLoss


class CombinedLoss(nn.Module):

    def __init__(self):
        super(CombinedLoss, self).__init__()

        self.wmr = 1
        self.wp = 10

        self.perceptual = PerceptualLoss()
        self.reconstruction = ReconstructionLoss()
    
    def forward(self, mask, y_pred, y_true):

        p_loss = self.perceptual(y_pred, y_true)
        mr_loss = self.reconstruction(mask, y_pred, y_true)

        loss = self.wmr * mr_loss + self.wp * p_loss

        return loss

