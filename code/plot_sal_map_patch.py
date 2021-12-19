import os
import sys
import glob
import time
import argparse
import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torchvision.transforms as tvtrans

from classification.utils import timeSince, printArgs, load_model, parse_model

from visualization.viz_helper import (
    get_first_conv_layer, get_last_conv_layer, viz_image_attr, normalize_image_attr, plot_figs, save_figs)

from sanity_check.utils import get_saliency_methods, get_saliency_masks


np.random.seed(2020)
torch.manual_seed(2020)


""" Usage
Analyze individual image patches
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
parser.add_argument('--group', type=str, default='baseline',
                    help='exp group')
opt = parser.parse_args()
printArgs(None, vars(opt))

ref_dataset_path = {
    'root_path': Path(opt.dataset_path),
    'train_filepath': Path(opt.dataset_path) / 'train_set.hdf5',
    'test_filepath': Path(opt.dataset_path) / 'test_set.hdf5',
}

model_para = parse_model(opt)
dataset_path = Path(opt.dataset_path) / '07-11-19_9dpi' / 'labelbox_data_12-20'
output_folder = Path(os.getcwd()).parent / 'results' / 'journal' / 'analyzer_patch_results' / opt.group

trays = ['tray5']
leaf_disk_image_filenames = ['139-4510009', '111-4509054', '217-4511022']


# Model
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
                                        gradcam=True,
                                        smooth_grad=True,
                                        deeplift=True)

# Green-Red color blindness
# default_cmap = 'gwr'
default_cmap = LinearSegmentedColormap.from_list(
    'MyColor', ['green', 'white', 'red']
)
label_class_map = {'0': 'Clear', '1': 'Infected'}

format_ = 'pdf'


for tray in trays:
    
    for leaf_disk_image_filename in leaf_disk_image_filenames:

        patch_filenames = [x for x in os.listdir(dataset_path / tray / leaf_disk_image_filename) if x.startswith(tray) and x.endswith('.png')]

        for patch_filename in patch_filenames:
            image_filepath = dataset_path / tray / leaf_disk_image_filename / patch_filename
            img = Image.open(image_filepath).resize((image_width, image_height))
            img_arr = np.asarray(img)

            input_img = preprocess(img_arr).unsqueeze(0).to(device)
            input_img.requires_grad = True

            logits = model(input_img)
            logtis_class = torch.argmax(logits, axis=1)[0].cpu().detach().item()

            if logtis_class == 1:

                output_masks = get_saliency_masks(saliency_methods, input_img, logtis_class)

                # for key, value in output_masks.items():
                #     print(f'{key} heatmap shape {value.shape}')

                # Visualization
                label = label_class_map[str(logtis_class)]
                # heatmap_figs, hist_figs = viz_image_attr(img_arr,
                #                                          output_masks,
                #                                          [],
                #                                          default_cmap,
                #                                          signs=["all", "all"],
                #                                          label=label,
                #                                          outlier_perc=1)
                abs_norm, no_abs_norm, norm_0_1 = normalize_image_attr(
                    img_arr, output_masks, hist=False)

                import copy
                heatmap_norm = copy.deepcopy(abs_norm)
                # Thresholding
                threshold = 0.2
                for key, value in heatmap_norm.items():
                    if key != 'Original':
                        value[value < threshold] = 0
                        value[value >= threshold] = 1
                        heatmap_norm.update({key: value})

                output_image_patch_folder = output_folder / f'{tray}_{leaf_disk_image_filename}'

                if not os.path.exists(output_image_patch_folder):
                    os.makedirs(output_image_patch_folder, exist_ok=True)

                # save_figs(output_image_patch_folder, patch_filename, heatmap_figs)
                # plot_figs(output_image_patch_folder,
                #         patch_filename,
                #         figs=[heatmap_norm],
                #         signs=['positive'],
                #         label=label,
                #         default_cmap='viridis',
                #         colorbar=True)

                # heatmap_norm.pop('Original')

                # fig, axs = plt.subplots(nrows=1, ncols=len(heatmap_norm)+3, figsize=(7.5, len(heatmap_norm)+3))
                fig, axs = plt.subplots(nrows=1, ncols=6)
                counter = 0

                for ax in axs:
                    ax.axis('off')

                axs[counter].imshow(heatmap_norm.pop('Original'))
                axs[counter].get_xaxis().set_visible(False)
                axs[counter].get_yaxis().set_visible(False)
                counter += 1 # Save one subplot for annotation
                axs[counter].imshow(np.zeros(img_arr.shape[:2]))
                axs[counter].get_xaxis().set_visible(False)
                axs[counter].get_yaxis().set_visible(False)
                counter += 1

                axs[counter].imshow(np.zeros(img_arr.shape[:2]))
                axs[counter].get_xaxis().set_visible(False)
                axs[counter].get_yaxis().set_visible(False)
                counter += 1 # Save one subplot for patch-based

                # # Gradient
                # axs[counter].imshow(heatmap_norm['Gradient'], vmin=0, vmax=1, cmap=default_cmap)
                # axs[counter].get_xaxis().set_visible(False)
                # axs[counter].get_yaxis().set_visible(False)
                # # axs[counter].set_title(key)
                # counter += 1    

                axs[counter].imshow(heatmap_norm['Gradient-SG'], vmin=0, vmax=1, cmap=default_cmap)
                axs[counter].get_xaxis().set_visible(False)
                axs[counter].get_yaxis().set_visible(False)
                # axs[counter].set_title(key)
                counter += 1    

                # GradCAM
                axs[counter].imshow(heatmap_norm['GradCAM'], vmin=0, vmax=1, cmap=default_cmap)
                axs[counter].get_xaxis().set_visible(False)
                axs[counter].get_yaxis().set_visible(False)
                # axs[counter].set_title(key)
                counter += 1    

                axs[counter].imshow(np.zeros(img_arr.shape[:2]))
                axs[counter].get_xaxis().set_visible(False)
                axs[counter].get_yaxis().set_visible(False)
                counter += 1

                # for key, value in heatmap_norm.items():
                #     axs[counter].imshow(value, vmin=0, vmax=1, cmap=default_cmap)
                #     axs[counter].get_xaxis().set_visible(False)
                #     axs[counter].get_yaxis().set_visible(False)
                #     # axs[counter].set_title(key)
                #     counter += 1      

                fig.subplots_adjust(wspace=0.1)
                patch_filename = os.path.splitext(patch_filename)[0] + f'.{format_}'
                plt.savefig(output_image_patch_folder / f'saliency_maps_{patch_filename}', format=format_,
                            dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()