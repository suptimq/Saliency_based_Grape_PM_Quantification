import os
import argparse
import h5py
import numpy as np
from pathlib import Path


""" Usage
Merge HDF5 files
"""

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='.')
parser.add_argument('--dirs', nargs='+',
                    help='Specify one or more directories')
parser.add_argument('--type', type=str, default='train')
opt = parser.parse_args()

main_folder = Path(opt.path)
dataset_folders = []

filename = 'train_set.hdf5' if opt.type == 'train' else 'test_set.hdf5'

output_filepath = main_folder / filename

for subfolder in opt.dirs:
    dataset_folders.append(main_folder / subfolder)

merged_dataset = h5py.File(output_filepath, 'w')
merged_img_arr = None
merged_label_arr = None

for idx, dataset_folder in enumerate(dataset_folders):
    dataset_filepath = dataset_folder / filename

    datast_hdf5 = h5py.File(dataset_filepath, 'r')
    dataset_img_arr = datast_hdf5['images'][:]
    dataset_label_arr = datast_hdf5['labels'][:]
    if idx == 0:
        merged_img_arr = dataset_img_arr
        merged_label_arr = dataset_label_arr
    else:
        merged_img_arr = np.concatenate(
            (merged_img_arr, dataset_img_arr))
        merged_label_arr = np.concatenate(
            (merged_label_arr, dataset_label_arr))
    datast_hdf5.close()

merged_dataset.create_dataset(name='images', data=merged_img_arr)
merged_dataset.create_dataset(name='labels', data=merged_label_arr)
print('HDF5 data file is saved as {}.'.format(output_filepath))
