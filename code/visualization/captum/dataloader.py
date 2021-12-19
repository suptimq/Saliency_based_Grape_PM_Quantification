import os
import torch
import torchvision
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class HyphalDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, target_class, train=False, transform=None, target_transform=None):
        self.root_dir = dataset_path['root_path']
        self.train_filepath = dataset_path['train_filepath']
        self.test_filepath = dataset_path['test_filepath']
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train is True:
            self.data_filepath = self.train_filepath
        else:
            self.data_filepath = self.test_filepath
        
        self.target_class = target_class

        with h5py.File(self.data_filepath, 'r') as f:
            image_ds = f['images'][:]
            label_ds = f['labels'][:]
            target_index = np.where(label_ds.squeeze() == target_class)[0]
            self.images = image_ds[target_index]
            self.labels = label_ds[target_index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cur_images = self.images[idx, :, :, :]
        cur_labels = self.labels[idx, :]
        cur_labels = cur_labels.squeeze().astype(dtype=np.int64)

        if self.transform is not None:
            cur_images = self.transform(cur_images)

        if self.target_transform is not None:
            cur_labels = self.target_transform(cur_labels)

        return cur_images, cur_labels