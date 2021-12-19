import os
import h5py
import argparse
from pathlib import Path
from PIL import Image

import pdb


def load_f5py(dataset_para):
    """
        Load data from HDF5 files or image directory
    """
    f = h5py.File(dataset_para['dataset_folder'] /
                  dataset_para['test_filepath'], 'r')
    image_ds = f['images']
    images = image_ds[:, ]
    label_ds = f['labels']
    labels = label_ds[:]
    return images, labels


""" Usage
Extract image patches specified by their index from the validation set

python save_image_patch.py 
    --dataset_path /Users/tim/Documents/Cornell/CAIR/BlackBird/Data/Hyphal_2020
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str,
                    required=True, help='path to data')
opt = parser.parse_args()

dataset_root_path = Path(opt.dataset_path) / 'data'

test_filepath = 'test_set.hdf5'

dataset_para = {
    'dataset_folder': dataset_root_path,
    'test_filepath': test_filepath,
}
print(opt)

# Import data from HDF5
images, labels = load_f5py(dataset_para)

# Positive: infected, negative: clear
t_p = [1333, 1234, 380, 1779, 1584, 1903, 1769, 570, 1545, 93]
f_p = [158, 492, 2071, 668, 1377, 892, 2232, 1265, 1373, 298]
t_n = [1667, 1338, 2150, 1902, 1669, 511, 877, 2579, 22, 958]
f_n = [674, 1643, 1078, 644, 138, 612, 361, 372, 1014, 695]

t_p_folder = dataset_root_path / 'true-infected'
f_p_folder = dataset_root_path / 'false-infected'
t_n_folder = dataset_root_path / 'true-clear'
f_n_folder = dataset_root_path / 'false-clear'

for idx in t_p:
    if not os.path.exists(t_p_folder):
        os.makedirs(t_p_folder, exist_ok=True)
    img_arr = images[idx]
    img = Image.fromarray(img_arr)
    img_filename = f'true_infected_{idx}.png'
    img_filepath = os.path.join(t_p_folder, img_filename)
    img.save(img_filepath)

for idx in f_p:
    if not os.path.exists(f_p_folder):
        os.makedirs(f_p_folder, exist_ok=True)
    img_arr = images[idx]
    img = Image.fromarray(img_arr)
    img_filename = f'false_infected_{idx}.png'
    img_filepath = os.path.join(f_p_folder, img_filename)
    img.save(img_filepath)


for idx in t_n:
    if not os.path.exists(t_n_folder):
        os.makedirs(t_n_folder, exist_ok=True)
    img_arr = images[idx]
    img = Image.fromarray(img_arr)
    img_filename = f'true_clear_{idx}.png'
    img_filepath = os.path.join(t_n_folder, img_filename)
    img.save(img_filepath)

for idx in f_n:
    if not os.path.exists(f_n_folder):
        os.makedirs(f_n_folder, exist_ok=True)
    img_arr = images[idx]
    img = Image.fromarray(img_arr)
    img_filename = f'false_clear_{idx}.png'
    img_filepath = os.path.join(f_n_folder, img_filename)
    img.save(img_filepath)
