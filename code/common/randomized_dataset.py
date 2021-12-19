import os
import h5py
import numpy as np

from pathlib import Path
from datetime import datetime


"""Usage
Shuffle a dataset
"""


root_path = Path(os.getcwd())
dataset_path = root_path / 'train_set.hdf5'

with h5py.File(dataset_path, 'r') as f:
    image_ds = f['images'][:]
    label_ds = f['labels'][:]

np.random.seed(1)
image_shuffler = np.random.permutation(len(image_ds))
np.random.seed(100)
label_shuffler = np.random.permutation(len(label_ds))

shuffled_image = image_ds[image_shuffler]
shuffled_label = label_ds[label_shuffler]

print('shuffled finished')

# shuffled_image.shape ----> (N, width, height, channel)
print(f'image: {shuffled_image.shape}')
print(f'label: {shuffled_label.shape}')

randomization_dataset = 'train_set_randomization.hdf5'

with h5py.File(randomization_dataset, 'w') as f:
    f.create_dataset(name='images', data=shuffled_image)
    f.create_dataset(name='labels', data=shuffled_label)
    f.attrs['year'] = datetime.now().year
    f.attrs['month'] = datetime.now().month
