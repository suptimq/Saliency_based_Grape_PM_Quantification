import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms as tvtrans

from analyzer_config import (
    CHANNELS, IMG_HEIGHT, IMG_WIDTH, IMG_EXT, INPUT_SIZE)

from metric import pixel_sr1, pixel_sr2, patch_sr

from utils import hard_thresholding, otsu_thresholding, apply_colormap_on_image, plot_colormap_on_image

from classification.inference import pred_img
from classification.utils import timeSince, printArgs, load_model, parse_model, set_logging

from analysis.leaf_mask import leaf_mask, on_focus

from visualization.viz_util import _normalize_image_attr
from visualization.viz_helper import (
    get_first_conv_layer, get_last_conv_layer, viz_image_attr, normalize_image_attr, plot_figs, save_figs)

from sanity_check.utils import get_saliency_methods, get_saliency_masks


np.random.seed(2020)


"""
Run Pearson's correlation at the leaf disk level
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
parser.add_argument('--group', type=str, default='saliency_based',
                    help='exp group')
parser.add_argument('--log', type=str, default='log',
                    help='log file path')
opt = parser.parse_args()

# Load manual analysis results
manual_analysis_filepath = 'Corelation_manualVsother.csv'
manual_analysis_df = pd.read_csv(manual_analysis_filepath)
sample_idx = manual_analysis_df['idx']
sample_id = manual_analysis_df['sample_id']

ref_dataset_path = {
    'root_path': Path(opt.dataset_path),
    'train_filepath': Path(opt.dataset_path) / 'train_set.hdf5',
    'test_filepath': Path(opt.dataset_path) / 'test_set.hdf5',
}

group = opt.group
image_timestamp = '07-08-19_6dpi'
dataset_path = Path(opt.dataset_path) / image_timestamp

output_csv_folder = Path(os.getcwd()).parents[1] / 'results' / 'leaf_correlation' / opt.group

if not os.path.exists(output_csv_folder):
    os.makedirs(output_csv_folder, exist_ok=True)

logger = set_logging(output_csv_folder / 'log.txt', 20)
logger.info(os.path.basename(__file__))
printArgs(logger, vars(opt))

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
                                    smooth_grad=True,
                                    gradcam=True)


# Write severity ratio as CSV files
META_COL_NAMES = ['timestamp', 'tray', 'filename',
                  'patch_sr', 'processing time'] + list(saliency_methods.keys())
severity_rate_df = pd.DataFrame(columns=META_COL_NAMES)
severity_rate_df_list = []
for th in pixel_th:
    severity_rate_df_list.append(pd.DataFrame(columns=META_COL_NAMES))


# Time
analysis_time = 0
total_time_masking = 0
total_time_crop_classify_saliency = 0
total_time_sr_calculation = 0

tray = 'tray1'
dataset_tray_path = dataset_path / tray

for sidx, sid in zip(sample_idx, sample_id):
    leaf_disk_image_filename = f'{sidx}-{sid}.tif'

    img_filepath = dataset_tray_path / leaf_disk_image_filename
    assert os.path.exists(img_filepath), f'{img_filepath} not found'

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

    imagename_text = os.path.splitext(leaf_disk_image_filename)[0]

    # Masking
    imask = leaf_mask(img, rel_th=rel_th)
    if imask is None:
        logger.info('Image: {}\tmasking ERROR'.format(imagename_text))
        continue
    imask = imask.astype('uint8') / 255

    # Timer breakpoint
    masking_done_time = time.time()
    logger.info('Finished loading mask: {}'.format(timeSince(start_time)))

    patch_idx = coor_x = coor_y = 0
    # Lost focused subimg
    infected_patch = clear_patch = discard_patch = lost_focus_patch = total_patch = 0
    infected_pixel = clear_pixel = discard_pixel = lost_focus_pixel = total_pixel = 0

    # Counter of each pixel
    counting_map = np.zeros(shape=(subim_height, subim_width))
    prob_attrs = np.zeros(
        shape=(subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float)

    saliency_attrs = {}
    for saliency_method_key in saliency_methods.keys():
        saliency_attrs[saliency_method_key] = np.zeros(
            shape=(subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float)

    # Crop
    for _ in range(subim_y):
        for _ in range(subim_x):
            subim_mask = imask[coor_y: coor_y + IMG_HEIGHT,
                               coor_x: coor_x + IMG_WIDTH]
            if not on_focus(subim_mask):
                # Set lost focused patches' pixel values as -inf
                lost_focus_patch += 1
                prob_attrs[patch_idx] = -np.inf
            else:
                # Cropping
                box = (coor_x, coor_y, coor_x +
                       IMG_WIDTH, coor_y + IMG_HEIGHT)
                subim = img.crop(box).resize((image_width, image_height))
                subim_arr = np.asarray(subim)

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
                            saliency_attrs[key][patch_idx] = F.interpolate(
                                val, (IMG_HEIGHT, IMG_WIDTH), mode='nearest')[0][0]
                        else:
                            saliency_attrs[key][patch_idx] = val

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

    # Reconstruction
    prob_heatmap = np.zeros(shape=(subim_height, subim_width), dtype=np.float)
    saliency_heatmaps = {}
    for key in saliency_methods.keys():
        saliency_heatmaps[key] = np.zeros(
            shape=(subim_height, subim_width), dtype=np.float)

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

    # Timer breakpoint
    crop_classify_saliency_done_time = time.time()
    logger.info('Finished crop and inference: {}'.format(
        timeSince(start_time)))

    # Severity rate calculation
    patch_info = {'infected_patch': infected_patch, 'clear_patch': clear_patch,
                  'discard_patch': discard_patch, 'lost_focus_patch': lost_focus_patch}
    heatmap_info = saliency_heatmaps.copy()
    heatmap_info['prob_heatmap'] = prob_heatmap
    threshold_info = {'patch_down_th': down_th,
                      'patch_up_th': up_th, 'pixel_th': pixel_th}

    severity_rate_patch, _ = patch_sr.metric(
        patch_info, heatmap_info, threshold_info)

    severity_rates_pixel, pixels_1 = pixel_sr1.metric(
        patch_info.copy(), heatmap_info.copy(), threshold_info.copy())

    # Timer breakpoint
    sr_calculation_done_time = time.time()

    for i, th in enumerate(pixel_th):
        record_df = pd.DataFrame(
            [[image_timestamp, tray, imagename_text, severity_rate_patch, sr_calculation_done_time - start_time] + list(severity_rates_pixel[th].values())], columns=META_COL_NAMES
        )
        severity_rate_df_list[i] = severity_rate_df_list[i].append(
            record_df, ignore_index=True)

    print(severity_rate_df_list[0])

    analysis_time += time.time() - start_time
    total_time_masking += masking_done_time - start_time
    total_time_crop_classify_saliency += crop_classify_saliency_done_time - masking_done_time
    total_time_sr_calculation += sr_calculation_done_time - \
        crop_classify_saliency_done_time

    logger.info('Analysis finished: {}'.format(timeSince(start_time)))
    logger.info('-------------------------------------------')

for i, th in enumerate(pixel_th):
    output_csv_folder_th = output_csv_folder / f'th_{th}'
    if not os.path.exists(output_csv_folder_th):
        os.makedirs(output_csv_folder_th, exist_ok=True)

    output_csv_filepath = output_csv_folder_th / 'severity_rate.csv'

    concat_keys = ['matlab_sr', 'Manual', 'Hypahl_transect']
    for concat_key in concat_keys:
        severity_rate_df_list[i][concat_key] = manual_analysis_df[concat_key][:len(
            severity_rate_df_list[i])]

    mean_row = severity_rate_df_list[i].mean(numeric_only=True, axis=0)
    std_row = severity_rate_df_list[i].std(numeric_only=True, axis=0)
    min_row = severity_rate_df_list[i].min(numeric_only=True, axis=0)
    max_row = severity_rate_df_list[i].max(numeric_only=True, axis=0)

    severity_rate_df_list[i] = severity_rate_df_list[i].append(
        mean_row, ignore_index=True)
    severity_rate_df_list[i] = severity_rate_df_list[i].append(
        std_row, ignore_index=True)
    severity_rate_df_list[i] = severity_rate_df_list[i].append(
        min_row, ignore_index=True)
    severity_rate_df_list[i] = severity_rate_df_list[i].append(
        max_row, ignore_index=True)

    severity_rate_df_list[i].to_csv(output_csv_filepath, index=False)
    logger.info('Saved {}'.format(output_csv_filepath))

logger.info(f'Average analysis time {analysis_time / len(sample_idx):.2f}')
logger.info(f'Average masking time {total_time_masking / len(sample_idx):.2f}')
logger.info(
    f'Average crop, classification, saliency time {total_time_crop_classify_saliency / len(sample_idx):.2f}')
logger.info(
    f'Average severity calculation time {total_time_sr_calculation / len(sample_idx):.2f}')


# Correlation
# from scipy.stats import pearsonr

# sr_results_df = manual_analysis_df[['matlab_sr', 'Manual', 'Hypahl_transect']]
# import pdb; pdb.set_trace()
# for key in saliency_methods.keys():
#     sr_results_df[key] = severity_rate_df[key]

# corr = sr_results_df.corr(method=lambda x, y: pearsonr(x, y)[0])
# pval = sr_results_df.corr(method=lambda x, y: pearsonr(x, y)[1])

# output_image_patch_folder = Path(
#     os.getcwd()) / 'tmp' / 'correlation_results'

# sr_results_df.to_csv(
#     output_image_patch_folder / 'value.csv', index=False)
# corr.to_csv(output_image_patch_folder / 'corr.csv', index=False)
# pval.to_csv(output_image_patch_folder / 'pval.csv', index=False)
