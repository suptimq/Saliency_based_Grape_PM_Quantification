import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn.functional as F
from torchvision import transforms as tvtrans

from analyzer_config import (
    CHANNELS, IMG_HEIGHT, IMG_WIDTH, IMG_EXT, INPUT_SIZE)

from metric import pixel_sr1, pixel_sr2, patch_sr


from classification.inference import pred_img
from classification.utils import timeSince, printArgs, load_model, parse_model, set_logging

from analysis.leaf_mask import leaf_mask, on_focus

from visualization.viz_util import _normalize_image_attr
from visualization.viz_helper import (
    get_first_conv_layer, get_last_conv_layer, viz_image_attr, normalize_image_attr, plot_figs, save_figs)

from sanity_check.utils import get_saliency_methods, get_saliency_masks


np.random.seed(2020)


""" Usage
Analyze the full-size leaf disc images and calculate the severity rate
Given a date, do analysis on all the data collected in that date
"""


parser = argparse.ArgumentParser()
parser.add_argument('--model_type',
                    default='VGG',
                    help='model used for training')
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model parameters')
parser.add_argument('--loading_epoch',
                    type=int,
                    required=True,
                    help='xth model loaded for inference')
parser.add_argument('--timestamp', required=True, help='model timestamp')
parser.add_argument('--outdim', type=int, default=2, help='number of classes')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--cuda_id',
                    default="0",
                    help='specify cuda id')
parser.add_argument('--target_class',
                    default="1",
                    help='class heatmap')
parser.add_argument('--step_size', type=int, default=224,
                    help='step size of sliding window')
parser.add_argument('--dataset_path', type=str, required=True,
                    help='root path to the data')
parser.add_argument('--model_path', type=str, required=True,
                    help='root path to the model')
parser.add_argument('--platform', type=str,  default='PMRobot',
                    help='robot platform')
parser.add_argument('--threshold', nargs='+',
                    help='thresholding value')
parser.add_argument('--log', type=str, default='../results/logs/random.log',
                    help='log file path')
parser.add_argument('--dpi', type=str, required=True,
                    help='inoculation date')
parser.add_argument('--group', type=str, default='baseline',
                    help='exp group')
parser.add_argument('--trays', nargs='+',
                    help='trays')
opt = parser.parse_args()

logger = set_logging(Path(str(opt.log)), 20)
logger.info(os.path.basename(__file__))
printArgs(logger, vars(opt))

ref_dataset_path = {
    'root_path': Path(opt.dataset_path),
    'train_filepath': Path(opt.dataset_path) / 'train_set.hdf5',
    'test_filepath': Path(opt.dataset_path) / 'test_set.hdf5',
}

image_timestamp = opt.dpi
dataset_path = Path(opt.dataset_path) / image_timestamp
mask_path = Path(opt.dataset_path) / f'{image_timestamp}_masking'
# output_folder = Path(
#     os.getcwd()).parent / 'results' / 'journal' / 'analyzer_leaf_results' / opt.group
output_folder = Path(opt.dataset_path).parents[1] / 'Stack_Visualiation' / image_timestamp


# Threshold for severity ratio
down_th = 0.2
up_th = 0.8
pixel_th = opt.threshold if opt.threshold else []

rel_th = 0.1 if opt.platform == 'BlackBird' else 0.2
target_class = int(opt.target_class) if opt.target_class != 'None' else None
step_size = opt.step_size

# Model
model_para = parse_model(opt)
model, device = load_model(model_para)
model.eval()
last_conv_layer = get_last_conv_layer(model)
first_conv_layer = get_first_conv_layer(model)
means = [118./255., 165./255., 92./255.]
stds = [40./255., 35./255., 51./255.]

# Input preprocessing transformation
if opt.model_type == 'Inception3':
    preprocess = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.Resize(299),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    image_width = image_height = 299
else:
    preprocess = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    image_width = image_height = 224

# Captum
saliency_methods = get_saliency_methods(model,
                                        last_conv_layer=last_conv_layer,
                                        first_conv_layer=first_conv_layer,
                                        ref_dataset_path=ref_dataset_path,
                                        image_width=image_width,
                                        transform=preprocess,
                                        device=device,
                                        partial=True,
                                        gradcam=True,
                                        gradient=True,
                                        smooth_grad=True,
                                        deeplift=True)


# Write severity ratio as CSV files
key = [f'{x}_sr2' for x in saliency_methods.keys()]
META_COL_NAMES = ['timestamp', 'tray', 'filename', 'patch_sr'] + key

# List all trays
# trays = [t for t in os.listdir(dataset_path) if t.startswith('tray') and not t.endswith('csv')]
trays = opt.trays

threshold = 0.2

# default_cmap = 'bwr'
default_cmap = LinearSegmentedColormap.from_list(
    'MyColor', ['green', 'white', 'red']
)

# Time
total_time = 0
total_time_2 = 0
format_ = 'pdf'
save_healthy = True

# Loop trays
for tray in trays:
    dataset_tray_path = dataset_path / tray
    # leaf_disk_image_filenames = ['4-4508004.tif', '21-4508024.tif','19-4508022.tif','268-Thompson_Seedless.tif']
    # leaf_disk_image_filenames = ['52-4508058.tif']
    # leaf_disk_image_filenames = ['268-Thompson_Seedless.tif']
    # leaf_disk_image_filenames = ['194-4510071.tif']
    leaf_disk_image_filenames = [x for x in os.listdir(dataset_tray_path) if x.endswith('.tif')]
 
    severity_rate_df_list = []
    for th in pixel_th:
        severity_rate_df_list.append(pd.DataFrame(columns=META_COL_NAMES))

    # Loop leaf disk images
    for leaf_disk_image_filename in leaf_disk_image_filenames:
        img_filepath = dataset_tray_path / leaf_disk_image_filename

        # Timer
        start_time = time.time()

        logger.info('-------------------------------------------')
        logger.info('Processing {} {} {}'.format(
            image_timestamp, tray, leaf_disk_image_filename))

        # Get info of resized image subim_x: number of patches one row
        img = Image.open(img_filepath)
        img_arr = np.asarray(img)
        width, height = img.size

        subim_x = (width - IMG_WIDTH) // step_size + 1
        subim_y = (height - IMG_HEIGHT) // step_size + 1
        subim_height = (subim_y - 1) * step_size + IMG_HEIGHT
        subim_width = (subim_x - 1) * step_size + IMG_WIDTH
        sub_img = img.crop((0, 0, subim_width, subim_height))
        sub_img_arr = np.asarray(sub_img)

        imagename_text = os.path.splitext(leaf_disk_image_filename)[0]

        # Masking
        imask = leaf_mask(img, rel_th=rel_th)
        if imask is None:
            logger.info('Image: {}\tmasking ERROR'.format(imagename_text))
            continue
        imask = imask.astype('uint8') / 255

        imask_filepath = mask_path / tray / f'{imagename_text}.json'
        # with open(imask_filepath, 'r') as f:
        #     invalid_patch = json.load(f)
        # invalid_patch_idx = list(invalid_patch.keys())
        
        t1 = time.time()
        logger.info('Finished loading mask: {}'.format(timeSince(start_time)))

        patch_idx = coor_x = coor_y = 0
        # Lost focused subimg
        infected_patch = clear_patch = discard_patch = lost_focus_patch = total_patch = 0
        infected_pixel = clear_pixel = discard_pixel = lost_focus_pixel = total_pixel = 0

        # Counter of each pixel
        counting_map = np.zeros(shape=(height, width))
        prob_attrs = np.zeros(
            shape=(subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float)

        saliency_attrs = {}
        for saliency_method_key in saliency_methods.keys():
            saliency_attrs[saliency_method_key] = np.zeros(
                shape=(subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float)

        f = os.path.splitext(leaf_disk_image_filename)[0]
        output_leaf_disk_image_folder = output_folder / f'{opt.dpi}_{tray}_{f}'
        if not os.path.exists(output_leaf_disk_image_folder):
            os.makedirs(output_leaf_disk_image_folder, exist_ok=True)

        # Crop
        for _ in range(subim_y):
            for _ in range(subim_x):
                subim_mask = imask[coor_y: coor_y + IMG_HEIGHT,
                                   coor_x: coor_x + IMG_WIDTH]
                if not on_focus(subim_mask):
                # if str(patch_idx) in invalid_patch_idx:
                    # Set lost focused patches' pixel values as -inf
                    lost_focus_patch += 1
                    prob_attrs[patch_idx] = -np.inf

                else:
                    # Cropping
                    box = (coor_x, coor_y, coor_x +
                           IMG_WIDTH, coor_y + IMG_HEIGHT)
                    subim = img.crop(box).resize((image_width, image_height))
                    subim_arr = np.asarray(subim)

                    # # Save image patches
                    # # plt.imsave(sub_img_arr,)
                    # output_leaf_disk_image_folder_saliency = output_leaf_disk_image_folder / 'image_patches'
                    # if not os.path.exists(output_leaf_disk_image_folder_saliency):
                    #     os.makedirs(
                    #         output_leaf_disk_image_folder_saliency, exist_ok=True)
                    # saved_patch_filepath = output_leaf_disk_image_folder_saliency / \
                    #     f'image_patch_{patch_idx}.{format_}'
                    # plt.imsave(saved_patch_filepath, subim_arr, format=format_, dpi=300)


                    # Preprocess
                    input_img = preprocess(subim_arr).unsqueeze(0).to(device)
                    input_img.requires_grad = True

                    # Get saliency maps
                    pred, prob = pred_img(input_img, model)
                    logtis_class = pred.cpu().detach().item()
                    prob_value = prob[0][1].cpu().detach().item()

                    if logtis_class == 1:
                        output_masks = get_saliency_masks(
                            saliency_methods, input_img, logtis_class, relu_attributions=True)

                        # Save to the entire array
                        prob_attrs[patch_idx] = prob_value
                        # Normalization
                        abs_norm, no_abs_norm, _0_1_norm = normalize_image_attr(
                            subim_arr, output_masks, hist=False)
                        abs_norm.pop('Original')

                        for key, val in abs_norm.items():
                            if image_height != IMG_HEIGHT:
                                # Adapt to the F.interpolate() API
                                val = torch.from_numpy(
                                    val[np.newaxis, np.newaxis, ...])
                                val = F.interpolate(
                                    val, (IMG_HEIGHT, IMG_WIDTH), mode='nearest')[0][0]
                                saliency_attrs[key][patch_idx] = val
                            else:
                                saliency_attrs[key][patch_idx] = val

                            # # Save patches' saliency maps
                            # if key == 'GradCAM':
                            #     output_leaf_disk_image_folder_saliency = output_leaf_disk_image_folder / key
                            #     if not os.path.exists(output_leaf_disk_image_folder_saliency):
                            #         os.makedirs(
                            #             output_leaf_disk_image_folder_saliency, exist_ok=True)
                            #     saved_patch_filepath = output_leaf_disk_image_folder_saliency / \
                            #         f'{key}_{patch_idx}.{format_}'
                            #     plt.imsave(saved_patch_filepath, val,
                            #             cmap=default_cmap, format=format_, dpi=300)

                    else:
                        if save_healthy:
                            output_leaf_disk_image_folder_saliency = output_leaf_disk_image_folder / 'healthy'
                            if not os.path.exists(output_leaf_disk_image_folder_saliency):
                                os.makedirs(
                                    output_leaf_disk_image_folder_saliency, exist_ok=True)
                            saved_patch_filepath = output_leaf_disk_image_folder_saliency / \
                                f'healthy.{format_}'
                            plt.imsave(saved_patch_filepath, np.zeros(
                                (224, 224)), cmap=default_cmap, format=format_, dpi=300)
                            save_healthy = False

                    # Increment the number of infected or clear patch
                    if prob_value >= up_th:
                        infected_patch += 1
                    elif prob_value <= down_th:
                        clear_patch += 1
                    else:
                        discard_patch += 1

                # Update pixel counter each loop to avoid ZeroDivisionError
                counting_map[coor_y: coor_y + IMG_HEIGHT,
                             coor_x: coor_x + IMG_WIDTH] += 1
                coor_x += step_size
                patch_idx += 1
            coor_x = 0
            coor_y += step_size

        counting_map[counting_map == 0] = 1

        logger.info('Finished crop and inference: {}'.format(
            timeSince(start_time)))

        # Reconstruction
        prob_heatmap = np.zeros(
            shape=(height, width), dtype=np.float)
        saliency_heatmaps = {}
        for key in saliency_methods.keys():
            saliency_heatmaps[key] = np.zeros(
                shape=(height, width), dtype=np.float)

        patch_idx = coor_x = coor_y = 0
        for _ in range(subim_y):
            for _ in range(subim_x):
                prob_heatmap[coor_y: coor_y + IMG_HEIGHT,
                             coor_x: coor_x + IMG_WIDTH] += prob_attrs[patch_idx]

                for key in saliency_methods.keys():
                    saliency_heatmaps[key][coor_y: coor_y + IMG_HEIGHT,
                                           coor_x: coor_x + IMG_WIDTH] += saliency_attrs[key][patch_idx]

                coor_x += step_size
                patch_idx += 1
            coor_x = 0
            coor_y += step_size

        # Divide by counting_map
        prob_heatmap = prob_heatmap / counting_map
        for key, val in saliency_heatmaps.items():
            saliency_heatmaps[key] = val / counting_map

        # Severity rate calculation
        patch_info = {'infected_patch': infected_patch, 'clear_patch': clear_patch,
                      'discard_patch': discard_patch, 'lost_focus_patch': lost_focus_patch}
        heatmap_info = saliency_heatmaps.copy()
        heatmap_info['prob_heatmap'] = prob_heatmap
        threshold_info = {'patch_down_th': down_th,
                          'patch_up_th': up_th, 'pixel_th': [threshold]}

        severity_rate_patch, pixels_patch = patch_sr.metric(
            patch_info, heatmap_info, threshold_info)
        severity_rates_pixel, pixels_1 = pixel_sr1.metric(patch_info.copy(), heatmap_info.copy(), threshold_info.copy())
        # severity_rates_pixel, pixels_2 = pixel_sr2.metric(
        #     patch_info.copy(), heatmap_info.copy(), threshold_info.copy())

        # Save severity rate related information
        output_sr_info_filepath = output_leaf_disk_image_folder / 'info.json'
        with open(output_sr_info_filepath, 'w') as fp:
            json.dump({'patch_based': severity_rate_patch, 'pixels': pixels_patch}, fp)
            json.dump(severity_rates_pixel, fp)
            json.dump(pixels_1[0], fp)
            json.dump(pixels_1[1], fp)

        # Visualization
        alpha = 0.7
        # Raw leaf disk
        output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
            f'{opt.dpi}_{f}_raw.{format_}'
        plt.imshow(img_arr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_leaf_disk_image_filepath, format=format_,
                    dpi=300, bbox_inches='tight', pad_inches=0)
        # Masked leaf disk
        output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
            f'{opt.dpi}_{f}_masked.{format_}'
        sub_img_arr_copy = img_arr.copy()
        sub_img_arr_copy[imask == 0] = 0
        sub_img_arr_copy = sub_img_arr_copy.astype('uint8') / 255
        plt.imshow(sub_img_arr_copy)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_leaf_disk_image_filepath, format=format_,
                    dpi=300, bbox_inches='tight', pad_inches=0)
        
        # Patch-based
        output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
            f'{opt.dpi}_{f}_patch_based.{format_}'
        value = prob_heatmap.copy()
        value[value < up_th] = 0
        value[value >= up_th] = 1
        value = value.astype('uint8')
        alphas = np.full(imask.shape, alpha)
        alphas[value == 0] = 0
        plt.imshow(value, alpha=alphas, cmap=default_cmap)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_leaf_disk_image_filepath, format=format_,
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Heatmap
        for key, value in saliency_heatmaps.items():
            value[value < threshold] = 0
            value[value >= threshold] = 1
            value = value.astype('uint8')

            # Overlap with original image
            output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
                f'{opt.dpi}_{key}_{f}_blended.{format_}'
            alphas = np.full(imask.shape, alpha)
            alphas[value == 0] = 0
            # plt.imshow(np.mean(sub_img_arr, axis=2) * imask, cmap='gray', alpha=0.5)
            plt.imshow(sub_img_arr_copy)
            plt.imshow(value, alpha=alphas, cmap=default_cmap)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_leaf_disk_image_filepath, format=format_,
                        dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.imshow(value, cmap=default_cmap)
            plt.axis('off')
            output_leaf_disk_image_filepath = output_leaf_disk_image_folder / \
                f'{opt.dpi}_{key}_{f}.{format_}'
            # Only heatmap
            plt.tight_layout()
            plt.savefig(output_leaf_disk_image_filepath, format=format_,
                        dpi=300, bbox_inches='tight', pad_inches=0)

        for i, th in enumerate(pixel_th):
            record_df = pd.DataFrame(
                [[image_timestamp, tray, imagename_text, severity_rate_patch] + list(severity_rates_pixel[float(th)].values())], columns=META_COL_NAMES
            )
            severity_rate_df_list[i] = severity_rate_df_list[i].append(
                record_df, ignore_index=True)

        total_time = total_time + time.time() - start_time
        total_time_2 = total_time_2 + time.time() - t1
        logger.info('Analysis finished: {}'.format(timeSince(start_time)))
        logger.info('-------------------------------------------')

    for i, th in enumerate(pixel_th):
        output_csv_folder_th = output_folder / f'th_{th}'
        if not os.path.exists(output_csv_folder_th):
            os.makedirs(output_csv_folder_th, exist_ok=True)

        output_csv_filepath = output_csv_folder_th / 'severity_rate.csv'

        severity_rate_df_list[i].to_csv(output_csv_filepath, index=False)
        logger.info('Saved {}'.format(output_csv_filepath))
