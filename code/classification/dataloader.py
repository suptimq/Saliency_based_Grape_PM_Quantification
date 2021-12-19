import os
from pathlib import Path
import torch
import torchvision
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class HyphalDataset(torch.utils.data.Dataset):

    label_class_map = {0: 'Non-infected', 1: 'Infected'}

    def __init__(self, dataset_path, train=True, transform=None, target_transform=None):
        self.root_dir = Path(dataset_path['root_path'])
        self.train_filepath = self.root_dir / dataset_path['train_filepath']
        self.test_filepath = self.root_dir / dataset_path['test_filepath']
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train is True:
            self.data_filepath = self.train_filepath
        else:
            self.data_filepath = self.test_filepath

        with h5py.File(self.data_filepath, 'r') as f:
            image_ds = f['images']
            self.images = image_ds[:, ]
            label_ds = f['labels']
            self.labels = label_ds[:]

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


def worker_init_fn(worker_id):
    print(torch.utils.data.get_worker_info())


# use to test the dataset class


def test_class():
    label_class_map = {0: 'Non-infected', 1: 'Infected'}

    # Parameters for dataset
    dataset_path = {
        'root_path': '/media/cornell/Data/tq42/Hyphal_2020/data',
        'meta_filepath': 'metadata.csv',
        'train_filepath': 'train_set.hdf5',
        'test_filepath': 'val_set.hdf5'
    }

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(299),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAffine(degrees=(0, 180), translate=(0.1, 0.1), scale=(0.8, 1.2)),
        # torchvision.transforms.RandomRotation(degrees=(0, 180)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ])

    hyphal_train_ds = HyphalDataset(
        dataset_path, train=True, transform=transform)

    hyphal_dl = torch.utils.data.DataLoader(hyphal_train_ds,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=2,
                                            worker_init_fn=worker_init_fn)
    data_iter = iter(hyphal_dl)
    for i in range(2):
        images, labels = data_iter.next()
        f, axarr = plt.subplots(2, 2)
        j = 0
        for row in range(2):
            for col in range(2):
                cur_img = images[j, ]
                cur_label = labels[j]
                axarr[row, col].imshow(np.transpose(cur_img, (1, 2, 0)))
                axarr[row, col].set_title('{0}'.format(
                    label_class_map[cur_label.item()]))
                j += 1
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'test_{i}.png')


if __name__ == "__main__":
    test_class()
