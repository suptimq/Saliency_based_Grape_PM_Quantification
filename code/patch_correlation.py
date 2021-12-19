import os
import sys
import glob
import time
import scipy
import argparse
import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

import torch
import torchvision.transforms as tvtrans

from utils import (hard_thresholding, otsu_thresholding,
                   skeletonization, IoU_metric, L2_metric)

from classification.utils import (
    timeSince, printArgs, load_model, parse_model, plot_confusion_matrix, set_logging)

from visualization.viz_helper import (
    get_first_conv_layer, get_last_conv_layer, viz_image_attr, normalize_image_attr, plot_figs, save_figs)

from sanity_check.utils import get_saliency_methods, get_saliency_masks


np.random.seed(2020)
torch.manual_seed(2020)


"""
Run Pearson's correlation at the image patch level
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
parser.add_argument('--dataset_path', type=str, required=True,
                    help='root path to the data')
parser.add_argument('--model_path', type=str, required=True,
                    help='root path to the model')
parser.add_argument('--threshold', nargs='+',
                    help='thresholding values')
parser.add_argument('--skel', action='store_true',
                    help='enable skeletonization for gradcam')
parser.add_argument('--group', type=str, default='saliency_based',
                    help='exp group')
parser.add_argument('--log', type=str, default='log',
                    help='log file path')
parser.add_argument('--exp', type=str, default='num_pixels',
                    help='experiment')
opt = parser.parse_args()

ref_dataset_path = {
    'root_path': Path(opt.dataset_path),
    'train_filepath': Path(opt.dataset_path) / 'train_set.hdf5',
    'test_filepath': Path(opt.dataset_path) / 'test_set.hdf5',
}

timestamp = '07-11-19_9dpi'
exp_type = opt.exp
dataset_path = Path(opt.dataset_path) / timestamp / 'labelbox_data_12-20'
gt_dataset_path = Path(opt.dataset_path) / timestamp / \
    f'labelbox_data_preprocessed_all_{exp_type}'

output_image_patch_folder = Path(os.getcwd()).parents[1] / 'results' / 'patch_correlation' / opt.group
if not os.path.exists(output_image_patch_folder):
    os.makedirs(output_image_patch_folder, exist_ok=True)

logger = set_logging(output_image_patch_folder / 'log.txt', 20)
logger.info(os.path.basename(__file__))
printArgs(logger, vars(opt))

trays = [x for x in os.listdir(dataset_path) if x.startswith('tray')]

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
                                        gradient=True,
                                        smooth_grad=True,
                                        gradcam=True)

label_class_map = {'0': 'Clear', '1': 'Infected'}

correct_infected_patch = 0
correct_clear_patch = 0
false_clear_patch = 0
false_infected_patch = 0

## {'0.1': {'Gradient': [], 'GradCAM': []}, '0.2': {'Gradient': [], 'GradCAM': []}}
num_infected_pixel = {}
true_label = []
pred_label = []

thresholds = opt.threshold if opt.threshold else []

# Loop trays
for tray in trays:
    dataset_tray_path = dataset_path / tray
    leaf_disk_image_filenames = [x for x in os.listdir(dataset_tray_path)]

    # Loop leaf disk images
    for leaf_disk_image_filename in leaf_disk_image_filenames:
        dataset_disk_path = dataset_tray_path / leaf_disk_image_filename
        patch_filenames = [x for x in os.listdir(
            dataset_disk_path) if x.startswith('tray')]

        logger.info('---------------------------------------------------------------')

        # Loop image patches
        for patch_filename in patch_filenames:
            image_filepath = dataset_path / tray / leaf_disk_image_filename / patch_filename

            logger.info(
                f'processing {tray} {leaf_disk_image_filename} {patch_filename}')

            # Load ground-truth mask
            gt_image_filepath = gt_dataset_path / tray / \
                leaf_disk_image_filename / patch_filename.replace('png', 'npy')

            assert os.path.exists(
                gt_image_filepath), 'annotation file not found'
            gt_mask = np.load(gt_image_filepath)
            gt_infected_pixel = len(gt_mask[gt_mask == 1])


            gt_class = 0
            if gt_infected_pixel:
                gt_class = 1
            else:
                gt_class = 0
            true_label.append(gt_class)

            # Load image patch
            img = Image.open(image_filepath).resize(
                (image_width, image_height))
            img_arr = np.asarray(img)
            height, width = img_arr.shape[1:]

            input_img = preprocess(img_arr).unsqueeze(0).to(device)
            input_img.requires_grad = True

            # Prediction
            logits = model(input_img)
            logtis_class = torch.argmax(logits, axis=1)[
                0].cpu().detach().item()
            pred_label.append(logtis_class)

            label = label_class_map[str(logtis_class)]
            if logtis_class == gt_class:
                # Calculate similarity only for infected class
                if logtis_class == 1:
                    output_masks = get_saliency_masks(
                        saliency_methods, input_img, logtis_class)
                    abs_norm, no_abs_norm, _0_1_norm = normalize_image_attr(
                        img_arr, output_masks, hist=False)
                    abs_norm.pop('Original')

                    # Loop thresholds
                    for th in thresholds:

                        for key, val in abs_norm.items():

                            if th == 'otsu':
                                saliency_mask = otsu_thresholding(
                                    val, vmin=0, vmax=1)
                            else:
                                saliency_mask = hard_thresholding(
                                    val, float(th), vmin=0, vmax=1)

                            # Post-process GradCAM
                            if key == 'GradCAM' and opt.skel:
                                saliency_mask = skeletonization(saliency_mask)

                            saliency_mask_infected_pixel = len(
                                saliency_mask[saliency_mask == 1])

                            if not num_infected_pixel.get(th, None):
                                num_infected_pixel[th] = {}
                            if not num_infected_pixel[th].get(key, None):
                                num_infected_pixel[th][key] = []
                            num_infected_pixel[th][key].append(
                                saliency_mask_infected_pixel)

                        if not num_infected_pixel[th].get('gt', None):
                            num_infected_pixel[th]['gt'] = []
                        num_infected_pixel[th]['gt'].append(gt_infected_pixel)

                    correct_infected_patch += 1
                else:
                    correct_clear_patch += 1
            else:
                if gt_class == 1 and logtis_class == 0:
                    false_clear_patch += 1
                    # img.save(output_image_patch_folder / patch_filename)
                else:
                    false_infected_patch += 1

for th in thresholds:

    output_image_patch_folder_th = output_image_patch_folder / f'th_{th}'
    if not os.path.exists(output_image_patch_folder_th):
        os.makedirs(output_image_patch_folder_th, exist_ok=True)

    # Pearson correlation
    num_infected_pixel_array = np.array(list(num_infected_pixel[th].values()))
    num_infected_pixel_df = pd.DataFrame(
        data=num_infected_pixel_array, index=list(num_infected_pixel[th].keys())).T
    corr = num_infected_pixel_df.corr(method=lambda x, y: pearsonr(x, y)[0])
    pval = num_infected_pixel_df.corr(method=lambda x, y: pearsonr(x, y)[1])

    num_infected_pixel_df.to_csv(
        output_image_patch_folder_th / 'value.csv', index=False)
    corr.to_csv(output_image_patch_folder_th / 'corr.csv', index=False)
    pval.to_csv(output_image_patch_folder_th / 'pval.csv', index=False)

# Confusion matrix
accuracy = 100.0 * (correct_infected_patch +
                    correct_clear_patch) / len(pred_label)

cm = confusion_matrix(true_label, pred_label)
f1 = f1_score(true_label, pred_label, average='macro')
plot_confusion_matrix(
    output_image_patch_folder, cm,
    list(label_class_map.values()),
    normalize=True,
    filename=f'confusion-matrix-{opt.model_type}-test-{opt.loading_epoch}.png',
    title=f'Confusion Matrix\nOverall Accuracy: {accuracy:.2f}%\nF1 Score: {f1:.2f}')
logger.info('Accuracy of the network on the {0} testing images: {1:.3f}%'.format(
    len(pred_label), accuracy))
