# Obtained from https://github.com/wolny/pytorch-3dunet
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

# from pytorch3dunet.unet3d.utils import expand_as_one_hot


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
        denominator = weight * denominator

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    
    return  (2 * intersect / denominator.clamp(min=epsilon))



class DiceLoss(nn.Module):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, ignore_index, weight=None):
        super(DiceLoss,self).__init__()
        self.ignore_index = ignore_index
        self.weight= weight

    def dice(self, predicted, target, weight):
        bs = target.size(0)
        num_classes = predicted.size(1)
        mask = target != self.ignore_index
        target = target.view(bs, -1)
        predicted = predicted.view(bs, num_classes, -1)
        target = F.one_hot((target[mask]).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
        target = target.permute(0,2,1) 
        predicted= predicted[mask]
#         predicted= predicted.permute(0,2,1) 
        return compute_per_channel_dice(predicted, target, weight=self.weight)
    
    def forward(self, predicted, target):
        # get probabilities from logits
        predicted =  nn.functional.softmax(predicted,dim=1)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(predicted, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
