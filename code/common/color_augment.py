import os
import PIL
import h5py

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as tvtrans

from pathlib import Path


"""Usage
Visualization of color factors after PyTorch color augmentation on a dataset
"""


main_folder = Path(os.getcwd())
dataset_folder = main_folder / 'NY84xPillans_18000' / 'dataset_12000'
# dataset_folder = main_folder
train_set_filepath = dataset_folder / 'train_set.hdf5'
val_set_filepath = dataset_folder / 'test_set.hdf5'

# Load data
with h5py.File(train_set_filepath, 'r') as f:
    image_ds = f['images']
    train_images = image_ds[:, ]


with h5py.File(val_set_filepath, 'r') as f:
    image_ds = f['images']
    val_images = image_ds[:, ]

# Color jitter
transform = tvtrans.Compose([
    tvtrans.ColorJitter(brightness=0.5, contrast=0.5, hue=0.1, saturation=0.1),
    # tvtrans.ColorJitter(brightness=[1.5, 1.7], contrast=[0.8, 1.0], hue=[0.05, 0.1]),
])

# Convert to HSV mode
# to_hsv = lambda x: np.asarray(PIL.Image.fromarray(x).convert('HSV'))


def to_hsv(x): return np.asarray(
    transform(PIL.Image.fromarray(x)).convert('HSV'))


train_images_hsv = np.array(list(map(to_hsv, train_images)))
val_images_hsv = np.array(list(map(to_hsv, val_images)))

# Get color factor


def hue(x): return np.mean(x[..., 0])
def saturation(x): return np.mean(x[..., 1])
def brightness(x): return np.mean(x[..., -1])
def contrast(x): return x[..., -1].max() - x[..., -1].min()


train_images_hue = np.array(list(map(hue, train_images_hsv)))
train_images_saturation = np.array(list(map(saturation, train_images_hsv)))
train_images_brightness = np.array(list(map(brightness, train_images_hsv)))
train_images_contrast = np.array(list(map(contrast, train_images_hsv)))

val_images_hue = np.array(list(map(hue, val_images_hsv)))
val_images_saturation = np.array(list(map(saturation, val_images_hsv)))
val_images_brightness = np.array(list(map(brightness, val_images_hsv)))
val_images_contrast = np.array(list(map(contrast, val_images_hsv)))

# Plot
fig, axs = plt.subplots(2, 4, figsize=(12, 8))
axs[0, 0].hist(train_images_brightness)
axs[0, 0].set_title('Train Set Brightness')

axs[0, 1].hist(train_images_hue)
axs[0, 1].set_title('Train Set Hue')

axs[0, 2].hist(train_images_saturation)
axs[0, 2].set_title('Train Set Saturation')

axs[0, 3].hist(train_images_contrast)
axs[0, 3].set_title('Train Set Contrast')

axs[1, 0].hist(val_images_brightness)
axs[1, 0].set_title('Val Set Brightness')

axs[1, 1].hist(val_images_hue)
axs[1, 1].set_title('Val Set Hue')

axs[1, 2].hist(val_images_saturation)
axs[1, 2].set_title('Val Set Saturation')

axs[1, 3].hist(val_images_contrast)
axs[1, 3].set_title('Val Set Contrast')

fig.suptitle('Color Distribution Transformed')

plt.savefig('color_distribution_train_val_set_trans.png')
