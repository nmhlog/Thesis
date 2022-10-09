import functools
import sys
import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../')
"""CHANGE THIS TO SOFTGROUP_OP"""
from lib.softgroup_ops.functions import voxelization
from util import cuda_cast, DiceLoss
from model.blocks import MLP, ResidualBlock, UNET,UNET_ASPP,UNET_ASPPv2,UNET_ATTN_ASPP


class semantic_segmentation_model(nn.Module):

    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 semantic_weight=None,
                 sem2ins_classes=[],
                 ignore_label=-100,
                 with_coords=True,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules=[],
                 modified_unet=None):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.semantic_weight = semantic_weight
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.with_coords = with_coords
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules
        self.modified_unet = modified_unet
        
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        in_channels = 6 if with_coords else 3
        
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1')).cuda()
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        """ ADDING UNET-ASPP and backbone differently"""
        is_ASPP =  getattr(self.modified_unet, 'ASPP', False) if self.modified_unet != None else False
        is_ASPPv2 =  getattr(self.modified_unet, 'ASPPv2', False) if self.modified_unet != None else False
        is_ATTN_ASPP =  getattr(self.modified_unet, 'ATTN_ASPP', False) if self.modified_unet != None else False
        if is_ASPP:
            self.unet = UNET_ASPP(block_channels,norm_fn,2).cuda()
        elif is_ASPPv2:
            self.unet = UNET_ASPPv2(block_channels, norm_fn, 2).cuda()
        elif is_ATTN_ASPP:
            self.unet = UNET_ATTN_ASPP(block_channels, norm_fn, 2).cuda()
        else:
            self.unet = UNET(block_channels, norm_fn, 2, block, indice_key_id=1).cuda()
        #self.unet = UNET(block_channels, norm_fn, 2, block, indice_key_id=1).cuda()
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU()).cuda()

        # point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2).cuda()


        self.init_weights()
        if self.semantic_weight:
            self.semantic_weight = torch.tensor(self.semantic_weight, dtype=torch.float, device='cuda')
        else:
            self.semantic_weight = None
            
        self.is_DICELOSS =  getattr(self.modified_unet, 'DICELOSS', True) if self.modified_unet != None else False
        if  self.is_DICELOSS:
            self.diceloss = DiceLoss( weight=self.semantic_weight,ignore_index=self.ignore_label).cuda()
            
        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()
        if not self.semantic_only:
            for m in [self.cls_linear, self.iou_score_linear]:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.forward_train(**batch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls,
                      pt_offset_labels, spatial_shape, batch_size, **kwargs):
        losses = {}
        if self.with_coords:
            #feats = coords_float
            feats = torch.cat((feats, coords_float), 1)
            
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores = self.forward_backbone(input, v2p_map)

        # point wise losses
        point_wise_loss = self.point_wise_loss(semantic_scores, semantic_labels)
        losses.update(point_wise_loss)
        
        # instance losses
        return self.parse_losses(losses)

    def point_wise_loss(self, semantic_scores, semantic_labels):
        losses = {}
        """
        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device='cuda')
        else:
            weight = None
        """
        if self.is_DICELOSS:
            semantic_loss = self.diceloss(semantic_scores, semantic_labels)
        else:
            semantic_loss = F.cross_entropy(
                semantic_scores, semantic_labels, weight=self.semantic_weight, ignore_index=self.ignore_label)

        losses['semantic_loss'] = semantic_loss

        return losses

    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses['loss'] = loss
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        return loss, losses

    @cuda_cast
    def forward_test(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                     scan_ids, **kwargs):
        color_feats = feats
        if self.with_coords:
            #feats = coords_float
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores = self.forward_backbone(input, v2p_map)
        semantic_preds = semantic_scores.max(1)[1]
        ret = dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            color_feats=color_feats.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy(),
            )

        return ret

    def forward_backbone(self, input, input_map):
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]
        semantic_scores = self.semantic_linear(output_feats)
        return semantic_scores
