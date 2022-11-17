import sys
sys.path.append('../')
from data.custom import CustomDataset
import math
import os.path as osp
from glob import glob
import sys
sys.path.append('../')
import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch
from torch.utils.data import Dataset

from lib.softgroup_ops.functions import voxelization_idx

class STPLS3DDataset(CustomDataset):

    CLASSES = ('building', 'low vegetation', 'med. vegetation', 'high vegetation', 'vehicle',
               'truck', 'aircraft', 'militaryVehicle', 'bike', 'motorcycle', 'light pole',
               'street sign', 'clutter', 'fence')
    
    def transform_test(self, xyz, rgb):
        xyz_middle = self.dataAugment(xyz, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        return xyz, xyz_middle, rgb

    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')
        data = self.load(filename)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        if data is None:
            return None
        xyz, xyz_middle, rgb = data
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(3) * 0.1
        return (scan_id, coord, coord_float, feat)

    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, coord_float, feat) = data

            scan_ids.append(scan_id)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            batch_id += 1
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)

        spatial_shape = np.clip(
            coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        return {
            'scan_ids': scan_ids, #name
            'coords': coords,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords, #Voxel Locs
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
        }