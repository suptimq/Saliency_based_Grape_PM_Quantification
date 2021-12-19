import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from pathlib import Path
from matplotlib.collections import PatchCollection


from utils import crop_image_patch


def bb_intersection_over_union(boxA, boxB):
    """
            Calculate the (x, y)-coordinates of the intersection rectangle
    Args:
        boxA and boxB: (x1, y1, x2, y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


timestamp = '07-11-19_9dpi'
leaf_disk_image_folder = Path(
    f'/media/cornell/Data/tq42/Hyphal_2020/data/{timestamp}')
output_folder = leaf_disk_image_folder / 'training_val_dataset_visualization'

annotation_folder = Path(
    '/media/cornell/Data/tq42/Hyphal_2020/data/annotation')
annotation_filenames = [x for x in os.listdir(
    annotation_folder) if x.endswith('csv') and ('9dpi' in x)]
print(annotation_filenames)

target_imagenames = ['111-4509054', '139-4510009', '217-4511022']

CHANNELS = 3
IMG_WIDTH = 224
IMG_HEIGHT = 224

COLS = ['timestamp', 'tray', 'filename',
        'position1', 'position2', 'overlapping']
overlapping_df = pd.DataFrame(columns=COLS)

for annotation_filename in annotation_filenames:
    annotation_filepath = annotation_folder / annotation_filename
    df = pd.read_csv(annotation_filepath)

    tray = os.path.splitext(annotation_filename)[0].split('_')[-1].lower()
    tray_folder = leaf_disk_image_folder / tray

    tray_output_folder = output_folder / tray
    if not os.path.exists(tray_output_folder):
        os.makedirs(tray_output_folder, exist_ok=True)

    for target_imagename in target_imagenames:
        leaf_disk_image_filepath = tray_folder / f'{target_imagename}.tif'

        patch_inds = df.loc[df['imagename'] ==
                            target_imagename, 'subindex'].values

        img = Image.open(leaf_disk_image_filepath)
        img_arr = np.asarray(img)
        width, height = img.size

        subim_x = width // IMG_WIDTH
        subim_y = height // IMG_HEIGHT
        horizontal_pos = height // 2
        vertical_pos = width // 2

        fig, ax = plt.subplots(1)
        ax.imshow(img_arr)
        draw_boxes = []
        draw_boxes_coors = []
        for patch_ind in patch_inds:
            box = crop_image_patch(patch_ind, subim_x)
            rect = patches.Rectangle((box[0], box[1]), IMG_WIDTH, IMG_HEIGHT)
            draw_boxes.append(rect)
            draw_boxes_coors.append(
                (box[0], box[1], box[0]+IMG_WIDTH, box[1]+IMG_HEIGHT))

        horizontal_boxes = []
        vertical_boxes = []
        horizontal_boxes_coors = []
        vertical_boxes_coors = []
        for i in range(subim_x):
            rect = patches.Rectangle(
                (i * IMG_WIDTH, horizontal_pos), IMG_WIDTH, IMG_HEIGHT)
            horizontal_boxes.append(rect)
            horizontal_boxes_coors.append(
                (i * IMG_WIDTH, horizontal_pos, i * IMG_WIDTH + IMG_WIDTH, horizontal_pos + IMG_HEIGHT))

        for i in range(subim_y):
            rect = patches.Rectangle(
                (vertical_pos, i * IMG_HEIGHT), IMG_WIDTH, IMG_HEIGHT)
            vertical_boxes.append(rect)
            vertical_boxes_coors.append(
                (vertical_pos, i * IMG_HEIGHT, vertical_pos + IMG_WIDTH, i * IMG_HEIGHT + IMG_HEIGHT))

        pc = PatchCollection(draw_boxes, facecolor='none',
                             alpha=0.5, edgecolor='r')
        pc_h = PatchCollection(horizontal_boxes, facecolor='none', alpha=0.5,
                               edgecolor='y')
        pc_v = PatchCollection(vertical_boxes, facecolor='none', alpha=0.5,
                               edgecolor='b')

        ax.add_collection(pc)
        ax.add_collection(pc_h)
        ax.add_collection(pc_v)

        threshold = 0.8
        for draw_boxes_coor in draw_boxes_coors:

            for coor in horizontal_boxes_coors:
                iou = bb_intersection_over_union(draw_boxes_coor, coor)
                if iou >= threshold:
                    tmp_df = pd.DataFrame(
                        [[timestamp, tray, target_imagename, draw_boxes_coor, coor, iou]], columns=COLS)
                    overlapping_df = overlapping_df.append(
                        tmp_df, ignore_index=True)

            for coor in vertical_boxes_coors:
                iou = bb_intersection_over_union(draw_boxes_coor, coor)
                if iou >= threshold:
                    tmp_df = pd.DataFrame(
                        [[timestamp, tray, target_imagename, draw_boxes_coor, coor, iou]], columns=COLS)
                    overlapping_df = overlapping_df.append(
                        tmp_df, ignore_index=True)

        fig.tight_layout()
        output_filepath = tray_output_folder / f'{target_imagename}.png'
        plt.savefig(output_filepath)

overlapping_df.to_csv(output_folder / 'overlapping.csv', index=False)
