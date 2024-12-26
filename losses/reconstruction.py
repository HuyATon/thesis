import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, mask, y_pred, y_true):
        """
        Compute reconstruction loss.
        Args:
        - mask (torch.Tensor): Binary mask tensor (B, C, H, W). [0, 1]
        - y_true (torch.Tensor): Ground truth image tensor (B, C, H, W).
        - y_pred (torch.Tensor): Predicted image tensor (B, C, H, W).
        """
        b = 1 - mask # => (0: hole, 1: valid)
        b = b.repeat(1, 3, 1, 1)

        term_1 = torch.abs(b * (y_true - y_pred)) / 10
        term_2 = torch.abs((1 - b) * (y_true - y_pred))

        loss = torch.mean(term_1 + term_2)
        return loss