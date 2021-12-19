import h5py
import numpy as np

from pathlib import Path
from datetime import datetime


"""Usage
Generate a balanced dataset
"""


np.random.seed(2020)

root_path = Path('./')
dataset_path = root_path / 'test_set.hdf5'

with h5py.File(dataset_path, 'r') as f:
    image_ds = f['images'][:]
    label_ds = f['labels'][:]
    filter_ = np.where(label_ds.squeeze() == 1)
    infected_images = image_ds[filter_[0]]
    filter_ = np.where(label_ds.squeeze() == 0)
    clear_images = image_ds[filter_[0]]

# infected_images.shape ----> (N, width, height, channel)
print(f'infected: {infected_images.shape}')
print(f'clear: {clear_images.shape}')

balanced_number = 550

new_infected_images = infected_images[:balanced_number]
new_clear_images = clear_images[:balanced_number]

new_dataset_images = np.concatenate(
    (new_infected_images, new_clear_images), axis=0)
new_dataset_labels = np.concatenate(
    (np.ones((balanced_number, 1)), np.zeros((balanced_number, 1))), axis=0)

print(f'new infected images: {new_dataset_images.shape}')
print(f'new infected labels: {new_dataset_labels.shape}')

shuffler = np.random.permutation(len(new_dataset_images))

shuffled_new_dataset_images = new_dataset_images[shuffler]
shuffled_new_dataset_labels = new_dataset_labels[shuffler]

balanced_dataset_path = root_path / 'balanced_test_set.hdf5'

with h5py.File(balanced_dataset_path, 'w') as f:
    f.create_dataset(name='images', data=shuffled_new_dataset_images)
    f.create_dataset(name='labels', data=shuffled_new_dataset_labels)
    f.attrs['year'] = datetime.now().year
    f.attrs['month'] = datetime.now().month
