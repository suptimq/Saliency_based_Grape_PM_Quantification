import os
import sys
import json
import time
import numpy as np

from PIL import Image
from pathlib import Path
from multiprocessing import Process

from analyzer_config import (
    CHANNELS, IMG_HEIGHT, IMG_WIDTH, IMG_EXT, INPUT_SIZE)

from classification.utils import timeSince

from analysis.leaf_mask import leaf_mask, on_focus

dpis = ['10_12_21_Nikita']
main_folder = Path('/media/cornell/Data/tq42/Hyphal_2020/data')

rel_th = 0.1
step_size = 224

def process_mask(dpi, start_time):
    dataset_path = main_folder / dpi

    trays = [x for x in os.listdir(dataset_path) if x.startswith('tray')]

    # Loop trays
    for tray in trays:

        output_json_tray_folder = main_folder / f'{dpi}_masking' / tray
        if not os.path.exists(output_json_tray_folder):
            os.makedirs(output_json_tray_folder)

        dataset_tray_path = dataset_path / tray
        leaf_disk_image_filenames = [x for x in os.listdir(
            dataset_tray_path) if x.endswith('.tif')]

        leaf_disk_image_filenames = sorted(leaf_disk_image_filenames, key=lambda  x: int(x.split('-')[0]))
        # Loop leaf disk images
        for leaf_disk_image_filename in leaf_disk_image_filenames:
            img_filepath = dataset_tray_path / leaf_disk_image_filename

            # Get info of resized image subim_x: number of patches one row
            img = Image.open(img_filepath)
            img_arr = np.asarray(img)
            width, height = img.size

            subim_x = (width - IMG_WIDTH) // step_size + 1
            subim_y = (height - IMG_HEIGHT) // step_size + 1
            subim_height = (subim_y - 1) * step_size + IMG_HEIGHT
            subim_width = (subim_x - 1) * step_size + IMG_WIDTH
            sub_img = img.crop((0, 0, subim_width, subim_height))

            imagename_text = os.path.splitext(leaf_disk_image_filename)[0]

            # Masking
            imask = leaf_mask(img, rel_th=rel_th)
            if imask is None:
                print('Image: {}\tmasking ERROR'.format(imagename_text))
                continue
            imask = imask.astype('uint8') / 255

            patch_idx = coor_x = coor_y = 0

            invalid_patch = {}

            # Crop
            for _ in range(subim_y):
                for _ in range(subim_x):
                    subim_mask = imask[coor_y: coor_y + IMG_HEIGHT,
                                    coor_x: coor_x + IMG_WIDTH]
                    if not on_focus(subim_mask):
                        invalid_patch[str(patch_idx)] = (coor_x, coor_y)
                    coor_x += step_size
                    patch_idx += 1
                coor_x = 0
                coor_y += step_size

            json_filepath = output_json_tray_folder / f'{imagename_text}.json'

            with open(json_filepath, 'w') as outfile: 
                json.dump(invalid_patch, outfile)

            print(f'{dpi} - {tray} - {leaf_disk_image_filename} usage time: {timeSince(start_time)}')

if __name__ == "__main__":
    jobs = []
    start_time = time.time()
    for dpi in dpis:
        p = Process(target=process_mask, args=(dpi, start_time))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()