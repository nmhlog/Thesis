import os.path as osp
from glob import glob

import numpy as np
import torch
import sys
sys.path.append('../')
from lib.softgroup_ops.functions import voxelization_idx
from data.custom import CustomDataset


class STPLS3DDataset_100(CustomDataset):

    CLASSES = ('building', 'low vegetation', 'med. vegetation', 'high vegetation', 'vehicle',
               'truck', 'aircraft', 'militaryVehicle', 'bike', 'motorcycle', 'light pole',
               'street sign', 'clutter', 'fence')
    # def load(self, filename):
    #     # TODO make file load results consistent
    #     xyz, rgb, semantic_label, instance_label = torch.load(filename)
    #     # subsample data
    #     if self.training:
    #         N = xyz.shape[0]
    #         inds = np.random.choice(N, int(N * 0.5), replace=False)
    #         xyz = xyz[inds]
    #         rgb = rgb[inds]
    #         semantic_label = semantic_label[inds]
    #         instance_label = self.getCroppedInstLabel(instance_label, inds)
    #     return xyz, rgb, semantic_label, instance_label
    
    def transform_test(self, xyz, rgb, semantic_label, instance_label):
        # devide into 4 piecies
        inds = np.arange(xyz.shape[0])
        piece_1 = inds[::4]
        piece_2 = inds[1::4]
        piece_3 = inds[2::4]
        piece_4 = inds[3::4]
        xyz_aug = self.dataAugment(xyz, False, False, False)

        xyz_list = []
        xyz_middle_list = []
        rgb_list = []
        semantic_label_list = []
        instance_label_list = []
        for batch, piece in enumerate([piece_1, piece_2, piece_3, piece_4]):
            xyz_middle = xyz_aug[piece]
            xyz = xyz_middle * self.voxel_cfg.scale
            xyz -= xyz.min(0)
            xyz_list.append(np.concatenate([np.full((xyz.shape[0], 1), batch), xyz], 1))
            xyz_middle_list.append(xyz_middle)
            rgb_list.append(rgb[piece])
            semantic_label_list.append(semantic_label[piece])
            instance_label_list.append(instance_label[piece])
        xyz = np.concatenate(xyz_list, 0)
        xyz_middle = np.concatenate(xyz_middle_list, 0)
        rgb = np.concatenate(rgb_list, 0)
        semantic_label = np.concatenate(semantic_label_list, 0)
        instance_label = np.concatenate(instance_label_list, 0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)  # TODO remove this
        return xyz, xyz_middle, rgb, semantic_label, instance_label
    
    def collate_fn(self, batch):
        if self.training:
            return super().collate_fn(batch)

        # assume 1 scan only
        (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num, inst_pointnum,
         inst_cls, pt_offset_label) = batch[0]
        scan_ids = [scan_id]
        coords = coord.long()
        batch_idxs = torch.zeros_like(coord[:, 0].int())
        coords_float = coord_float.float()
        feats = feat.float()
        semantic_labels = semantic_label.long()
        instance_labels = instance_label.long()
        instance_pointnum = torch.tensor([inst_pointnum], dtype=torch.int)
        instance_cls = torch.tensor([inst_cls], dtype=torch.long)
        pt_offset_labels = pt_offset_label.float()
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0],
                                None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, 4)
        return {
            'scan_ids': scan_ids,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'pt_offset_labels': pt_offset_labels,
            'spatial_shape': spatial_shape,
            'batch_size': 4
        }
    
    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # ignore instance of class 0 and reorder class id
        instance_cls = [x - 1 if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label
