import os
import PIL
import h5py

import numpy as np

import torchvision.transforms as tvtrans

from pathlib import Path


"""Usage
Calculate mean and std of a dataset
"""


main_folder = Path(os.getcwd())
# dataset_folder = main_folder / 'NY84xPillans_18000' / 'dataset_12000'
dataset_folder = main_folder
train_set_filepath = dataset_folder / 'train_set.hdf5'
val_set_filepath = dataset_folder / 'test_set.hdf5'

# Load data
with h5py.File(train_set_filepath, 'r') as f:
    image_ds = f['images']
    train_images = image_ds[:, ]


with h5py.File(val_set_filepath, 'r') as f:
    image_ds = f['images']
    val_images = image_ds[:, ]


train_images_red = train_images[..., 0]
train_images_green = train_images[..., 1]
train_images_blue = train_images[..., 2]

val_images_red = val_images[..., 0]
val_images_green = val_images[..., 1]
val_images_blue = val_images[..., 2]

train_images_mean = (np.mean(train_images_red), np.mean(
    train_images_green), np.mean(train_images_blue))
train_images_std = (np.std(train_images_red), np.std(
    train_images_green), np.std(train_images_blue))

val_images_mean = (np.mean(val_images_red), np.mean(
    val_images_green), np.mean(val_images_blue))
val_images_std = (np.std(val_images_red), np.std(
    val_images_green), np.std(val_images_blue))

print(f'{train_images.shape[0]} training samples')
print(f'train mean {train_images_mean} std {train_images_std}')

print(f'{val_images.shape[0]} val samples')
print(f'val mean {val_images_mean} std {val_images_std}')
