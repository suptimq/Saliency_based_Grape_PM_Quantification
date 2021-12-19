import os
import cv2
import h5py
import random
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

np.random.seed(2021)

# Parameters for dataset
dataset_path = [    '/media/cornell/Data/tq42/Hyphal_2020/data/segmentation/cls_seg_mask',
    'train_set_DeepLift.hdf5',
    'train_set_SG.hdf5',
    'train_set_GradCAM.hdf5']

images = []
masks = []

for i in range(1, 4):
    filepath = Path(dataset_path[0]) / dataset_path[i]
    with h5py.File(filepath, 'r') as f:
        image_ds = f['images']
        images.append(image_ds[:,])
        mask_ds = f['labels']
        masks.append(mask_ds[:,])

f, axs = plt.subplots(4, 5)
counter = 0

for i in range(4):

    mask1 = masks[0][i]
    mask2 = masks[1][i]
    mask3 = masks[2][i]
    mask_ensemble = mask1 + mask2 + mask3
    mask_ensemble_copy = mask_ensemble.copy()
    mask_ensemble_copy[mask_ensemble_copy < 3] = 0

    axs[counter][0].imshow(images[0][i])
    axs[counter][1].imshow(mask1, cmap='Reds')
    axs[counter][2].imshow(mask2, cmap='Reds')
    axs[counter][3].imshow(mask3, cmap='Reds')
    axs[counter][4].imshow(mask_ensemble_copy, cmap='Reds')

    counter = counter + 1

plt.tight_layout()
plt.savefig('test.png')