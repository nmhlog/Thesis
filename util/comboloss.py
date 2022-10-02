from pickle import NONE
from tkinter.messagebox import NO
from typing_extensions import Self
from diceloss import DiceLoss
import torch
import torch.nn.functional as F
from torch import nn as nn

class ComboCEDiceLoss(nn.Module):
    """
        Combination CrossEntropy (BCE) and Dice Loss with an optional running mean and loss weighing.
    """

    def __init__(self,semantic_weight=None,ignore_label=None,use_running_mean=False, bce_weight=1, dice_weight=1, eps=1e-6, gamma=0.9, combined_loss_only=True, **_):
        """
        :param use_running_mean: - bool (default: False) Whether to accumulate a running mean and add it to the loss with (1-gamma)
        :param bce_weight: - float (default: 1.0) Weight multiplier for the BCE loss (relative to dice)
        :param dice_weight: - float (default: 1.0) Weight multiplier for the Dice loss (relative to BCE)
        :param eps: -
        :param gamma:
        :param combined_loss_only: - bool (default: True) whether to return a single combined loss or three separate losses
        """

        super().__init__()
        '''
        Note: BCEWithLogitsLoss already performs a torch.sigmoid(pred)
        before applying BCE!
        '''
        self.semantic_weight= semantic_weight if semantic_weight else None
        self.ignore_label= ignore_label if ignore_label else None
        
        self.cross_entropy =  nn.cross_entropy(weight=self.semantic_weight, ignore_index=self.ignore_label)
        self.diceloss = DiceLoss( weight=self.semantic_weight,ignore_index=self.ignore_label)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma
        self.combined_loss_only = combined_loss_only

        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.use_running_mean is True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()
            
    def reset_parameters(self):
        self.running_bce_loss.zero_()
        self.running_dice_loss.zero_()
    def to(self, device):
        super().to(device=device)
        self.bce_logits_loss.to(device=device)
        self.diceloss.to(device=device)
        
    def forward(self, outputs, labels, **_):
        # inputs and targets are assumed to be BxCxWxH (batch, color, width, height)
        bce_loss = self.cross_entropy(outputs, labels)
        dice_loss = self.diceloss(outputs, labels)
        # dice_target = (labels == 1).float()
        # dice_output = torch.sigmoid(outputs)
        # intersection = (dice_output * dice_target).sum()
        # union = dice_output.sum() + dice_target.sum() + self.eps
        # dice_loss = (-torch.log(2 * intersection / union))

        if self.use_running_mean is False:
            bmw = self.bce_weight
            dmw = self.dice_weight
            # loss += torch.clamp(1 - torch.log(2 * intersection / union),0,100)  * self.dice_weight
        else:
            self.running_bce_loss = self.running_bce_loss * self.gamma + bce_loss.data * (1 - self.gamma)
            self.running_dice_loss = self.running_dice_loss * self.gamma + dice_loss.data * (1 - self.gamma)

            bm = float(self.running_bce_loss)
            dm = float(self.running_dice_loss)

            bmw = 1 - bm / (bm + dm)
            dmw = 1 - dm / (bm + dm)

        loss = bce_loss * bmw + dice_loss * dmw

        if self.combined_loss_only:
            return loss
        else:
            return loss, bce_loss, dice_loss


