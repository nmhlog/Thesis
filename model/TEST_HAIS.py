from dataclasses import is_dataclass
import functools

from yaml import scan

import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from lib.hais_ops.functions import hais_ops
from util import cuda_cast, force_fp32, rle_encode, DiceLoss
from model.blocks import MLP, ResidualBlock, UNET,UNET_ASPP,UNET_ASPPv2,UNET_ATTN_ASPP

""" 
DELETE cls 
UPDATE DATA CLUSTRING PARTS
AND MAKE SURE THE OPS PART

"""

class HAIS(nn.Module):

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
                 hais_util=None,
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
        self.hais_util = hais_util
        self.modified_unet = modified_unet
        

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        
        # backbone
        in_channels = 6 if with_coords else 3
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        "ASPP flag"
        is_ASPP =  getattr(self.modified_unet, 'ASPP', False) if self.modified_unet != None else False
        is_ASPPv2 =  getattr(self.modified_unet, 'ASPPv2', False) if self.modified_unet != None else False
        is_ATTN_ASPP =  getattr(self.modified_unet, 'ATTN_ASPP', False) if self.modified_unet != None else False
        if is_ASPP:
            self.unet = UNET_ASPP(block_channels,norm_fn,2).cuda()
        elif is_ASPPv2:
            self.unet = UNET_ASPPv2(block_channels, norm_fn, 2).cuda()
        elif is_ATTN_ASPP:
            self.unet = UNET_ATTN_ASPP(block_channels, norm_fn, 2,is_asppv1=is_ASPP).cuda()
        else:
            self.unet = UNET(block_channels, norm_fn, 2, block, indice_key_id=1).cuda()
        
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU()).cuda()

        # point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2).cuda()
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2).cuda()

        # topdown refinement path
        if not semantic_only:
            self.tiny_unet = UNET([channels, 2 * channels], norm_fn, 2, block, indice_key_id=11)
            self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
#             self.cls_linear = nn.Linear(channels, instance_classes + 1)
            self.mask_linear = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, 1))
            self.iou_score_linear = nn.Linear(channels,  1) #score

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
            for m in [self.iou_score_linear]:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch,current_epoch,return_loss=False,test=False):
        if return_loss:
            return self.forward_train(current_epoch,**batch)
        elif not test:
            return self.forward_test(current_epoch,**batch)
        else :
            return self.forward_test_without_val(current_epoch,**batch)

    @cuda_cast
    def forward_train(self, epoch, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls,
                      pt_offset_labels, spatial_shape, batch_size,scan_ids ,**kwargs):
        losses = {}
        # print(scan_ids)
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = hais_ops.voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map)
        #return semantic_scores, pt_offsets, output_feats     
        # point wise losses
        # Semantic segmentation and centroid loss.
        point_wise_loss = self.point_wise_loss(semantic_scores, pt_offsets, semantic_labels,
                                               instance_labels, pt_offset_labels)
        losses.update(point_wise_loss)
        
        # instance losses
        if not self.semantic_only:
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                    batch_idxs, coords_float)
            if proposals_offset.shape[0] > self.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]
            inst_feats, inst_map = self.clusters_voxelization(
                proposals_idx,
                proposals_offset,
                output_feats,
                coords_float,
                rand_quantize=True,
                **self.instance_voxel_cfg)
            
            iou_scores, mask_scores = self.forward_instance(inst_feats, inst_map,proposals_offset,epoch)
            instance_loss = self.instance_loss( mask_scores, iou_scores, proposals_idx,
                                               proposals_offset, instance_labels, instance_pointnum,epoch,scan_ids)
            losses.update(instance_loss)
        return self.parse_losses(losses)

    def point_wise_loss(self, semantic_scores, pt_offsets, semantic_labels, instance_labels,
                        pt_offset_labels):
        losses = {}
                
        if self.is_DICELOSS:
            semantic_loss = self.diceloss(semantic_scores, semantic_labels)
        else:
            semantic_loss = F.cross_entropy(
                semantic_scores, semantic_labels, weight=self.semantic_weight, ignore_index=self.ignore_label)
        losses['semantic_loss'] = semantic_loss

        pos_inds = instance_labels != self.ignore_label
        if pos_inds.sum() == 0:
            offset_loss = 0 * pt_offsets.sum()
        else:
            offset_loss = F.l1_loss(
                pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
        losses['offset_loss'] = offset_loss
        return losses

    @force_fp32(apply_to=('mask_scores', 'iou_scores'))
    def instance_loss(self, mask_scores, iou_scores, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum,epoch,scan_ids=None):
        losses = {}
        mask_scores_sigmoid = torch.sigmoid(mask_scores)

        if getattr(self.hais_util, 'cal_iou_based_on_mask', False) \
                and (epoch > self.hais_util.cal_iou_based_on_mask_start_epoch):
            ious, mask_label =  hais_ops.cal_iou_and_masklabel(proposals_idx[:, 1].cuda(), \
                proposals_offset.cuda(), instance_labels, instance_pointnum, mask_scores_sigmoid.detach(), 1)
        else:
            ious, mask_label =  hais_ops.cal_iou_and_masklabel(proposals_idx[:, 1].cuda(), \
                proposals_offset.cuda(), instance_labels, instance_pointnum, mask_scores_sigmoid.detach(), 0)
        # ious: (nProposal, nInstance)
        # mask_label: (sumNPoint, 1)
        try:
            mask_label_weight = (mask_label != -1).float()
        except:
            torch.save(mask_label,"error {}.pth".format(epoch))
            print(mask_label)
            print(scan_ids)
        finally:
            mask_label_weight = (mask_label != -1).float()
        mask_label[mask_label==-1.] = 0.5 # any value is ok
        mask_loss = torch.nn.functional.binary_cross_entropy(mask_scores_sigmoid, mask_label, weight=mask_label_weight, reduction='none')
        mask_loss = mask_loss.mean()
        #losses['mask_loss'] = (mask_loss, mask_label_weight.sum())
        losses['mask_loss'] = mask_loss
        
        gt_ious, _ = ious.max(1)  # gt_ious: (nProposal) float, long
            

        gt_scores = self.get_segmented_scores(gt_ious, self.hais_util.fg_thresh, self.hais_util.bg_thresh)

        score_loss = F.binary_cross_entropy(torch.sigmoid(iou_scores.view(-1)), gt_scores, reduction ="none")
        score_loss = score_loss.mean()

        #losses['score_loss'] = (score_loss, gt_ious.shape[0])
        losses['score_loss'] = score_loss
    
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
    def forward_test(self, epoch,batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                     scan_ids, **kwargs):
        color_feats = feats
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = hais_ops.voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(
            input, v2p_map,x4_split=self.test_cfg.x4_split)
        if self.test_cfg.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            semantic_labels = self.merge_4_parts(semantic_labels)
            instance_labels = self.merge_4_parts(instance_labels)
            pt_offset_labels = self.merge_4_parts(pt_offset_labels)
        semantic_preds = semantic_scores.max(1)[1]
        ret = dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            color_feats=color_feats.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy(),
            offset_preds=pt_offsets.cpu().numpy(),
            offset_labels=pt_offset_labels.cpu().numpy(),
            instance_labels=instance_labels.cpu().numpy()
            )
        if not self.semantic_only:
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                    batch_idxs, coords_float,'test')
            inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset,
                                                              output_feats, coords_float,
                                                              **self.instance_voxel_cfg)
            iou_scores, mask_scores = self.forward_instance(inst_feats, inst_map,proposals_offset,epoch)
            pred_instances = self.get_instances(scan_ids[0], proposals_offset,proposals_idx, semantic_scores, iou_scores, mask_scores,N=feats.shape[0])
            gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
            ret.update(dict(pred_instances=pred_instances, gt_instances=gt_instances))
        return ret
    
    @cuda_cast
    def forward_test_without_val(self, epoch,batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats, spatial_shape, batch_size, scan_ids, **kwargs):
        color_feats = feats
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = hais_ops.voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(
            input, v2p_map,x4_split=self.test_cfg.x4_split)
        if self.test_cfg.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            semantic_labels = self.merge_4_parts(semantic_labels)
            instance_labels = self.merge_4_parts(instance_labels)
            pt_offset_labels = self.merge_4_parts(pt_offset_labels)
        semantic_preds = semantic_scores.max(1)[1]
        ret = dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            color_feats=color_feats.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            # semantic_labels=semantic_labels.cpu().numpy(),
            offset_preds=pt_offsets.cpu().numpy()
            # offset_labels=pt_offset_labels.cpu().numpy(),
            # instance_labels=instance_labels.cpu().numpy()
            )
        if not self.semantic_only:
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                    batch_idxs, coords_float,'test')
            inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset,
                                                              output_feats, coords_float,
                                                              **self.instance_voxel_cfg)
            iou_scores, mask_scores = self.forward_instance(inst_feats, inst_map,proposals_offset,epoch)
            pred_instances = self.get_instances(scan_ids[0], proposals_offset,proposals_idx, semantic_scores, iou_scores, mask_scores,N=feats.shape[0])
            # gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
            ret.update(dict(pred_instances=pred_instances))
        return ret
    
    def forward_backbone(self, input, input_map, x4_split=False):
        if x4_split:
            output_feats = self.forward_4_parts(input, input_map)
            output_feats = self.merge_4_parts(output_feats)
        else:
            output = self.input_conv(input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]

        semantic_scores = self.semantic_linear(output_feats)
        pt_offsets = self.offset_linear(output_feats)
        return semantic_scores, pt_offsets, output_feats
    def forward_4_parts(self, x, input_map):
        """Helper function for s3dis: devide and forward 4 parts of a scene."""
        outs = []
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1)
            out = self.input_conv(x_new)
            out = self.unet(out)
            out = self.output_layer(out)
            outs.append(out.features)
        outs = torch.cat(outs, dim=0)
        return outs[input_map.long()]

    def merge_4_parts(self, x):
        """Helper function for s3dis: take output of 4 parts and merge them."""
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]
        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new  

    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self,semantic_scores,pt_offsets,batch_idxs,coords_float,training_mode='train'):

        batch_size = batch_idxs.max() + 1
        semantic_preds = semantic_scores.max(1)[1]

        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        object_idxs = torch.nonzero(semantic_preds > 0).view(-1) # floor idx 0, wall idx 1
 
        batch_idxs_ = batch_idxs[object_idxs]
        batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
        coords_ = coords_float[object_idxs]
        pt_offsets_ = pt_offsets[object_idxs]
        semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()
        
        idx, start_len = hais_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                           radius, mean_active)
        if training_mode =="train":
            using_set_aggr = getattr(self.hais_util, 'using_set_aggr_in_training', True)
        else:
            using_set_aggr = getattr(self.hais_util, 'using_set_aggr_in_testing', True)
        proposals_idx, proposals_offset = hais_ops.hierarchical_aggregation(semantic_preds_cpu, (coords_ + pt_offsets_).cpu(), 
                                                                            idx.cpu(), start_len.cpu(), batch_idxs_.cpu(), 
                                                                            training_mode, using_set_aggr)             
        proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()        

        return proposals_idx, proposals_offset

    def forward_instance(self, inst_feats, inst_map,proposals_offset,epoch):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)
        score_feats = feats.features[inst_map.long()]

        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]

        if getattr(self.hais_util, 'use_mask_filter_score_feature', False)  and \
                    epoch > self.hais_util.use_mask_filter_score_feature_start_epoch:
            mask_index_select = torch.ones_like(mask_scores)
            mask_index_select[torch.sigmoid(mask_scores) < self.hais_util.mask_filter_score_feature_thre] = 0.
            score_feats = score_feats * mask_index_select
        score_feats = hais_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
        iou_scores = self.iou_score_linear(score_feats)  # (nProposal, 1)
#         iou_scores = self.iou_score_linear(feats)

        return iou_scores, mask_scores

    @force_fp32(apply_to=('semantic_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self, scan_id, 
                      proposals_offset, 
                      proposals_idx, 
                      semantic_scores, 
                      iou_scores, 
                      mask_scores,
                      N,
                      semantic_label_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        # num_instances = cls_scores.size(0)
        # num_points = semantic_scores.size(0)
        # cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        scores_pred = torch.sigmoid(iou_scores.view(-1))
        proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device)
        test_mask_score_thre = self.test_cfg.mask_score_thr #getattr(cfg, 'test_mask_score_thre', -0.5)
        _mask = mask_scores.squeeze(1) > test_mask_score_thre
        proposals_pred[proposals_idx[_mask][:, 0].long(), proposals_idx[_mask][:, 1].long()] = 1
        ## semantic_label_idx => semantic label 
        semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long
        # semantic_id_idx = semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]
        score_mask = (scores_pred > self.test_cfg.cls_score_thr) #cfg.TEST_SCORE_THRESH
        scores_pred = scores_pred[score_mask]
        proposals_pred = proposals_pred[score_mask]
        semantic_id = semantic_id[score_mask]
                # semantic_id_idx = semantic_id_idx[score_mask]

                # npoint threshold
        proposals_pointnum = proposals_pred.sum(1)
        npoint_mask = (proposals_pointnum >= self.test_cfg.min_npoint) #cfg.TEST_NPOINT_THRESH)
        clusters = scores_pred[npoint_mask]
        cluster_scores = proposals_pred[npoint_mask]
        cluster_semantic_id = semantic_id[npoint_mask]
        # nclusters = clusters.shape[0]
        cls_pred = cluster_semantic_id.cpu().numpy()
        score_pred = clusters.cpu().detach().numpy()
        mask_pred = cluster_scores.cpu().numpy()
        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = hais_ops.sec_min(coords, clusters_offset.cuda())
        coords_max = hais_ops.sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        """
        NOTE 
        BELLOW SAME LIKE HAIS*
        """
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = hais_ops.voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = hais_ops.voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
        return voxelization_feats, inp_map

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets
    
    def get_segmented_scores(self,scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh + 1e-5)
        b = bg_thresh / (bg_thresh - fg_thresh + 1e-5)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores
