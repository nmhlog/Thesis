import torch
import tqdm
from random import sample
import numpy as np
import glob

def random_sample(coords: np.ndarray, colors: np.ndarray, semantic_labels: np.ndarray,
                  instance_labels: np.ndarray, ratio: float):
    num_points = coords.shape[0]
    num_sample = int(num_points * ratio)
    sample_ids = sample(range(num_points), num_sample)

    # downsample
    coords = coords[sample_ids]
    colors = colors[sample_ids]
    semantic_labels = semantic_labels[sample_ids]
    instance_labels = instance_labels[sample_ids]

    return coords, colors, semantic_labels, instance_labels

if __name__ == "__main__":
    data_folder = "val_100_100"
    files = sorted(glob.glob(data_folder + '/*.pth'))
    print('processing: {data_folder}')
    for data_file in tqdm.tqdm(files):
    # for data_file in glob.glob(osp.join(data_dir, "*.pth")):
        (coords, colors, semantic_labels, instance_labels) = torch.load(data_file)
        coords, colors, semantic_labels, instance_labels = random_sample(coords, colors, semantic_labels, instance_labels, 0.5)
        torch.save((coords, colors, semantic_labels, instance_labels), data_file)
    
    
    data_folder = "train_100_100"
    files = sorted(glob.glob(data_folder + '/*.pth'))
    print('processing: {data_folder}')
    for data_file in tqdm.tqdm(files):
    # for data_file in glob.glob(osp.join(data_dir, "*.pth")):
        (coords, colors, semantic_labels, instance_labels) = torch.load(data_file)
        coords, colors, semantic_labels, instance_labels = random_sample(coords, colors, semantic_labels, instance_labels, 0.5)
        torch.save((coords, colors, semantic_labels, instance_labels), data_file)
    
    data_folder = "train_100_50"
    files = sorted(glob.glob(data_folder + '/*.pth'))
    print('processing: {data_folder}')
    for data_file in tqdm.tqdm(files):
    # for data_file in glob.glob(osp.join(data_dir, "*.pth")):
        (coords, colors, semantic_labels, instance_labels) = torch.load(data_file)
        coords, colors, semantic_labels, instance_labels = random_sample(coords, colors, semantic_labels, instance_labels, 0.5)
        torch.save((coords, colors, semantic_labels, instance_labels), data_file)
