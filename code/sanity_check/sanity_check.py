#!/usr/bin/env python
# coding: utf-8
import os
import sys
import PIL
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import spearmanr as spr
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from skimage import feature

from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torchvision.transforms as tvtrans
from torchvision import models


from utils import (VisualizeImageDiverging, VisualizeImageGrayscale,
                   LoadImage, ShowImage,
                   normalize_image, abs_grayscale_norm, attribute_image_features, diverging_norm,
                   init_weights, reinit,
                   get_layer_randomization_order, get_first_conv_layer, get_last_conv_layer, get_saliency_methods, get_saliency_masks)

from utils import (printArgs, load_model, parse_model, load_f5py)


np.random.seed(1701)
torch.manual_seed(1701)
torch.cuda.manual_seed(1701)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 6
plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['lines.markersize'] = 0.2

axis_font_prop = {'fontname': 'Arial',
                  'fontsize': 8, 'weight': 'bold', 'ha': 'right'}
axis_label_font_prop = {'fontname': 'Arial',
                  'fontsize': 8, 'weight': 'bold', 'ha': 'center'}
txt_font_prop = {'fontname': 'Arial', 'fontsize': 10,
                 'weight': 'bold', 'ha': 'center', 'backgroundcolor': 'w'}
marker_font_prop = {'fontname': 'Arial',
                    'fontsize': 12, 'weight': 'bold', 'ha': 'center'}
infigure_font_prop = {'fontname': 'Arial', 'fontsize': 7,
                      'weight': 'bold', 'ha': 'center', 'va': 'bottom'}


parser = argparse.ArgumentParser()
parser.add_argument('--model_type',
                    default='SqueezeNet',
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
parser.add_argument('--mode', type=str, default='cascading',
                    choices=['cascading', 'independent'],
                    help='cascading or independent')

opt = parser.parse_args()
printArgs(None, vars(opt))

# Load leaf disease class
imagenet_class_index = {"0": "Clear", "1": "Infected"}

model_para = parse_model(opt)
model, device = load_model(model_para)
model.eval()
last_conv_layer = get_last_conv_layer(model)
first_conv_layer = get_first_conv_layer(model)

# Get leaf images
ref_dataset_path = {
    'root_path': Path(opt.dataset_path),
    'train_filepath': Path(opt.dataset_path) / 'train_set.hdf5',
    'test_filepath': Path(opt.dataset_path) / 'val_set.hdf5',
}

images, labels = load_f5py(ref_dataset_path)

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

dataset_path = Path(opt.dataset_path) / '07-11-19_9dpi' / 'labelbox_data_12-20'

tray = 'tray1'
leaf_disk_image_filename = '139-4510009'
patch_filename = 'tray1_139-4510009_Vertical_patch_14_Infected.png'
f = patch_filename.split('.')[0]
image_filepath = dataset_path / tray / leaf_disk_image_filename / patch_filename

saved_filepath = Path(os.getcwd()).parents[0] / 'results' / 'journal' / 'sanity_check' / 'partial'
saved_filepath_model = saved_filepath / opt.mode / opt.group / opt.model_type / f'{f}'

if not os.path.exists(str(saved_filepath_model)):
    os.makedirs(str(saved_filepath_model), exist_ok=True)

img = PIL.Image.open(image_filepath).resize((image_width, image_height))
demo_img = np.asarray(img)
demo_batch = [demo_img]

input_img = preprocess(demo_img).unsqueeze(0).to(device)
input_img.requires_grad = True

logits = model(input_img)
logtis_class = torch.argmax(logits, axis=1)[0].cpu().detach().item()

format_ = 'pdf'

# plt.imshow(demo_img)
# plt.title(f'Prediction {logtis_class}')
# plt.axis('off')
# plt.savefig(saved_filepath_model / 'raw_image.png')
plt.imsave(saved_filepath_model /
           f'raw_image.{format_}', demo_img, format=format_, dpi=300)

saliency_methods = get_saliency_methods(model,
                                        last_conv_layer=last_conv_layer,
                                        first_conv_layer=first_conv_layer,
                                        ref_dataset_path=ref_dataset_path,
                                        image_width=image_width,
                                        transform=preprocess,
                                        partial=True,
                                        gradient=True,
                                        smooth_grad=True,
                                        gradcam=True,
                                        guided_backprop=True,
                                        deeplift=True,
                                        explanation_map=True,
                                        device=device)

output_masks = get_saliency_masks(
    saliency_methods, input_img, logtis_class, relu_attributions=False)

for key, value in output_masks.items():
    print(f'{key} heatmap shape {value.shape}')

# We are now doing normalization of the different attributions
# normalize by absolute values
list_of_masks_abs_norm = []

# normalize but keep the signs
list_of_masks_no_abs_norm = []
list_of_masks_0_1_norm = []

new_dict_abs_norm = {}
new_dict_no_abs_norm = {}
new_dict_0_1_norm = {}
for key in output_masks:
    mask = output_masks[key]
    mask_abs_norm = abs_grayscale_norm(mask)
    mask_no_abs_norm = diverging_norm(mask)
    mask_0_1_norm = normalize_image(mask)
    new_dict_abs_norm[key] = mask_abs_norm
    new_dict_no_abs_norm[key] = mask_no_abs_norm
    new_dict_0_1_norm[key] = mask_0_1_norm
list_of_masks_abs_norm.append(new_dict_abs_norm)
list_of_masks_no_abs_norm.append(new_dict_no_abs_norm)
list_of_masks_0_1_norm.append(new_dict_0_1_norm)


# combine all the images to be plotted into one long list
# format is [(input, mask1, mask2, etc)]
mask_order = list(saliency_methods.keys()) + ['GBP-GC']
# mask_order = list(saliency_methods.keys())
master_mask_list_abs_norm = []
master_mask_list_no_abs_norm = []
for i, (img_dict_abs_norm, img_dict_no_abs_norm) in enumerate(zip(list_of_masks_abs_norm,
                                                                  list_of_masks_no_abs_norm)):
    # first append original image
    og_img = demo_batch[i]
    master_mask_list_abs_norm.append(og_img)
    master_mask_list_no_abs_norm.append(og_img)

    # loop through dicts to append each mask type
    for mask_type in mask_order:
        master_mask_list_abs_norm.append(img_dict_abs_norm[mask_type])
        master_mask_list_no_abs_norm.append(img_dict_no_abs_norm[mask_type])


# for demo in demo_batch:
#     print(demo.shape)
master_cascading_randomization_dictionary = {}  # key will be layer name.
layer_randomization_order = get_layer_randomization_order(model, opt.mode)

vgg16_layers = ['classifier', 'features/28', 'features/26', 'features/24',
                'features/21', 'features/19', 'features/17', 'features/14', 'features/12', 'features/10',
                'features/7', 'features/5', 'features/2', 'features/0']

resnet50_layers = ['fc', 'layer4', 'layer3', 'layer2', 'layer1', 'conv1']

inception3_layers = ['fc', 'Mixed_7c',
                     'Mixed_7b', 'Mixed_7a',
                     'Mixed_6e', 'Mixed_6d',
                     'Mixed_6c', 'Mixed_6b',
                     'Mixed_6a', 'Mixed_5d',
                     'Mixed_5c', 'Mixed_5b',
                     'Conv2d_4a_3x3', 'Conv2d_3b_1x1',
                     'Conv2d_2b_3x3', 'Conv2d_2a_3x3',
                     'Conv2d_1a_3x3']

if opt.model_type == 'VGG':
    mapping_layers = vgg16_layers
elif opt.model_type == 'ResNet':
    mapping_layers = resnet50_layers
else:
    mapping_layers = inception3_layers

# begin randomization
layer_randomization_cls_result = [1]
for i, layer_name in enumerate(layer_randomization_order):
    if opt.mode == 'cascading':
        print("Cascading reinitialization up to on layer {}".format(layer_name))
    else:
        print("Independent reinitialization on layer {}".format(layer_name))

    # list of parameters to be reintialized
    if opt.mode == 'cascading':
        index = mapping_layers.index(layer_name)
        layer_list = mapping_layers[:index+1]
    else:
        layer_list = [layer_name]

    torch.cuda.empty_cache()

    # load a new model.
    model, device = load_model(model_para)
    model.eval()
    last_conv_layer = get_last_conv_layer(model)
    first_conv_layer = get_first_conv_layer(model)

    # reinitialize all trainable ops up to that layer.
    reinit(model, layer_list)

    saliency_methods = get_saliency_methods(model,
                                            last_conv_layer=last_conv_layer,
                                            first_conv_layer=first_conv_layer,
                                            ref_dataset_path=ref_dataset_path,
                                            image_width=image_width,
                                            transform=preprocess,
                                            partial=True,
                                            gradient=True,
                                            gradcam=True,
                                            smooth_grad=True,
                                            guided_backprop=True,
                                            deeplift=True,
                                            explanation_map=True,
                                            device=device)

    # list to store collection of images
    list_of_random_mask_per_layer = []

    # model predictions for the images
    logits = model(input_img)
    logtis_class = torch.argmax(logits, axis=1)[0].cpu().detach().item()
    print(f'Prediction class {logtis_class} true class 1')
    layer_randomization_cls_result.append(logtis_class)

    output_masks = get_saliency_masks(
        saliency_methods, input_img, logtis_class, relu_attributions=False)
    list_of_random_mask_per_layer.append(output_masks)

    master_cascading_randomization_dictionary[layer_name] = list_of_random_mask_per_layer

master_cascading_randomization_dictionary_abs_norm = {}
master_cascading_randomization_dictionary_no_abs_norm = {}
master_cascading_randomization_dictionary_0_1_norm = {}

for layer in master_cascading_randomization_dictionary:
    mask_list_abs_norm = []
    mask_list_no_abs_norm = []
    mask_list_0_1_norm = []
    for i, mask_dict in enumerate(master_cascading_randomization_dictionary[layer]):
        # first append original image
        new_dict_abs_norm = {}
        new_dict_no_abs_norm = {}
        new_dict_0_1_norm = {}
        for key in mask_dict:
            mask = mask_dict[key]

            mask_abs_norm = abs_grayscale_norm(mask)
            mask_no_abs_norm = diverging_norm(mask)
            mask_0_1_norm = normalize_image(mask)

            new_dict_abs_norm[key] = mask_abs_norm
            new_dict_no_abs_norm[key] = mask_no_abs_norm
            new_dict_0_1_norm[key] = mask_0_1_norm

        mask_list_abs_norm.append(new_dict_abs_norm)
        mask_list_no_abs_norm.append(new_dict_no_abs_norm)
        mask_list_0_1_norm.append(new_dict_0_1_norm)
    master_cascading_randomization_dictionary_abs_norm[layer] = mask_list_abs_norm
    master_cascading_randomization_dictionary_no_abs_norm[layer] = mask_list_no_abs_norm
    master_cascading_randomization_dictionary_0_1_norm[layer] = mask_list_0_1_norm

for key in master_cascading_randomization_dictionary:
    assert len(master_cascading_randomization_dictionary[key]) == len(
        master_cascading_randomization_dictionary_no_abs_norm[key])
    for i, item in enumerate(master_cascading_randomization_dictionary[key]):
        assert len(item) == len(
            master_cascading_randomization_dictionary_no_abs_norm[key][i])
for key in master_cascading_randomization_dictionary:
    assert len(master_cascading_randomization_dictionary[key]) == len(
        master_cascading_randomization_dictionary_abs_norm[key])
    for i, item in enumerate(master_cascading_randomization_dictionary[key]):
        assert len(item) == len(
            master_cascading_randomization_dictionary_abs_norm[key][i])
for key in master_cascading_randomization_dictionary:
    assert len(master_cascading_randomization_dictionary[key]) == len(
        master_cascading_randomization_dictionary_0_1_norm[key])
    for i, item in enumerate(master_cascading_randomization_dictionary[key]):
        assert len(item) == len(
            master_cascading_randomization_dictionary_0_1_norm[key][i])

cascading_master_plotting_list_abs_norm = []
cascading_master_plotting_list_no_abs_norm = []

index_to_input_to_plot = 0  # this must be less than the len of demo_batch


mask_order = list(saliency_methods.keys()) + ['GBP-GC']
# mask_order = list(saliency_methods.keys())
for method in mask_order:
    # first insert normal saliency method
    normal_mask_abs_norm = list_of_masks_abs_norm[index_to_input_to_plot][method]
    normal_mask_no_abs_norm = list_of_masks_no_abs_norm[index_to_input_to_plot][method]
    cascading_master_plotting_list_abs_norm.append(normal_mask_abs_norm)
    cascading_master_plotting_list_no_abs_norm.append(normal_mask_no_abs_norm)
    for layer in layer_randomization_order:
        mask_abs_norm = master_cascading_randomization_dictionary_abs_norm[
            layer][index_to_input_to_plot][method]
        cascading_master_plotting_list_abs_norm.append(mask_abs_norm)

        mask_no_abs_norm = master_cascading_randomization_dictionary_no_abs_norm[
            layer][index_to_input_to_plot][method]
        cascading_master_plotting_list_no_abs_norm.append(mask_no_abs_norm)

print(len(cascading_master_plotting_list_abs_norm))
print(len(cascading_master_plotting_list_no_abs_norm))

ncols = len(layer_randomization_order) + 1  # plus one for original mask
nrows = len(mask_order)
figsize = (7, nrows)

# converts normalized image into 0-255 range for
# plotting.
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(nrows, ncols,
                       wspace=0.0, hspace=0.0)
cmap = 'gray'
count = 0
titles = list(saliency_methods.keys()) + ['GBP-GC']
# titles = list(saliency_methods.keys())
new_layer_names = ['Normal\nModel']

for val in layer_randomization_order:
    if val != 'classifier' and len(val) > 9:
        module = val[:8]
        layer = val[9:]
        val = f'{module}\n{layer}'
    new_layer_names.append(val)

for i in range(nrows):
    for j in range(ncols):
        ax = plt.subplot(gs[i, j])
        ax.imshow(cascading_master_plotting_list_abs_norm[count],
                  vmin=0.0,
                  vmax=1.0,
                  cmap=cmap)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # add labels
        if count < ncols:
            # adjust title(s)
            if len(new_layer_names[count]) > 8:
                pad_size = 12
            else:
                pad_size = 12
            # ax.set_title(new_layer_names[count], fontsize=7, rotation=90,
            #              pad=pad_size)

        # increment count
        count += 1

        if ax.is_first_col():
            ax.set_ylabel(titles[i], fontsize=7,
                          rotation='horizontal', ha='right')
plt.savefig(saved_filepath_model / f'{opt.mode}_randomization_gray.{format_}',
            format=format_, dpi=300, bbox_inches='tight', pad_inches=0)

# converts normalized image into 0-255 range for
# plotting.
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(nrows, ncols,
                       wspace=0.0, hspace=0.0)
cmap = 'hsv'
cmap2 = LinearSegmentedColormap.from_list(
    "Health", ["blue", "white", "green"]
)
count = 0
for i in range(nrows):
    for j in range(ncols):
        cmap_ = cmap if layer_randomization_cls_result[j] else cmap2
        ax = plt.subplot(gs[i, j])
        ax.imshow(cascading_master_plotting_list_abs_norm[count],
                  vmin=-1.0,
                  vmax=1.0,
                  cmap=cmap)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # add labels
        if count < ncols:
            # adjust title(s)
            if len(new_layer_names[count]) > 8:
                pad_size = 12
            else:
                pad_size = 12
            # ax.set_title(new_layer_names[count], fontsize=7, rotation=90,
            #              pad=pad_size)

        # increment count
        count += 1

        if ax.is_first_col():
            ax.set_ylabel(titles[i], fontsize=7,
                          rotation='horizontal', ha='right')
plt.savefig(saved_filepath_model / f'{opt.mode}_randomization_bwr.{format_}',
            format=format_, dpi=300, bbox_inches='tight', pad_inches=0)

# converts normalized image into 0-255 range for
# plotting.
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(nrows, ncols,
                       wspace=0.0, hspace=0.0)
cmap = 'coolwarm'
count = 0
for i in range(nrows):
    for j in range(ncols):
        ax = plt.subplot(gs[i, j])
        ax.imshow(cascading_master_plotting_list_no_abs_norm[count],
                  vmin=0.0,
                  vmax=1.0,
                  cmap=cmap)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # add labels
        if count < ncols:
            # adjust title(s)
            if len(new_layer_names[count]) > 8:
                pad_size = 12
            else:
                pad_size = 12
            # ax.set_title(new_layer_names[count], fontsize=7, rotation=90,
            #              pad=pad_size)

        # increment count
        count += 1

        if ax.is_first_col():
            ax.set_ylabel(titles[i], fontsize=7,
                          rotation='horizontal', ha='right')
plt.savefig(saved_filepath_model / f'{opt.mode}_randomization_coolwarm.{format_}',
            format=format_, dpi=300, bbox_inches='tight', pad_inches=0)

# converts normalized image into 0-255 range for
# plotting.
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(nrows, ncols,
                       wspace=0.0, hspace=0.0)
cmap = 'coolwarm'
count = 0
for i in range(nrows):
    for j in range(ncols):
        ax = plt.subplot(gs[i, j])
        ax.imshow(cascading_master_plotting_list_no_abs_norm[count],
                  vmin=-1.0,
                  vmax=1.0,
                  cmap=cmap)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # add labels
        if count < ncols:
            # adjust title(s)
            if len(new_layer_names[count]) > 8:
                pad_size = 12
            else:
                pad_size = 12
            # ax.set_title(new_layer_names[count], fontsize=7, rotation=90,
            #              pad=pad_size)

        # increment count
        count += 1

        if ax.is_first_col():
            ax.set_ylabel(titles[i], fontsize=7,
                          rotation='horizontal', ha='right')
plt.savefig(saved_filepath_model / f'{opt.mode}_randomization_pos_neg.{format_}',
            format=format_, dpi=300, bbox_inches='tight', pad_inches=0)


methods_list = list(
    master_cascading_randomization_dictionary_abs_norm[layer_randomization_order[0]][0].keys())

# dictionary to save all of the metrics.
rank_correlation_dictionary_abs_norm = {}
rank_correlation_dictionary_no_abs_norm = {}
ssim_dictionary_0_1_norm = {}
hog_dictionary_0_1_norm = {}

# initialize the dictionaries appropriately.
for layer in layer_randomization_order:
    rank_correlation_dictionary_abs_norm[layer] = {}
    rank_correlation_dictionary_no_abs_norm[layer] = {}
    ssim_dictionary_0_1_norm[layer] = {}
    hog_dictionary_0_1_norm[layer] = {}
    for method in methods_list:
        rank_correlation_dictionary_abs_norm[layer][method] = []
        rank_correlation_dictionary_no_abs_norm[layer][method] = []
        ssim_dictionary_0_1_norm[layer][method] = []
        hog_dictionary_0_1_norm[layer][method] = []

for layer in master_cascading_randomization_dictionary_abs_norm:
    for i, mask_dict in enumerate(master_cascading_randomization_dictionary_abs_norm[layer]):
        for method in methods_list:
            normal_mask_abs_norm = list_of_masks_abs_norm[i][method]
            normal_mask_no_abs_norm = list_of_masks_no_abs_norm[i][method]
            normal_mask_0_1_norm = list_of_masks_0_1_norm[i][method]

            rand_mask_abs_norm = mask_dict[method]
            rand_mask_no_abs_norm = master_cascading_randomization_dictionary_no_abs_norm[
                layer][i][method]
            rand_mask_0_1_norm = master_cascading_randomization_dictionary_0_1_norm[
                layer][i][method]

            # compute rank correlation
            rk_abs_abs_value_norm, _ = spr(
                normal_mask_abs_norm.flatten(), rand_mask_abs_norm.flatten())
            rk_no_abs_value_norm, _ = spr(
                normal_mask_no_abs_norm.flatten(), rand_mask_no_abs_norm.flatten())

            # compute ssim
            ss1 = ssim(normal_mask_0_1_norm, rand_mask_0_1_norm,
                       gaussian_weights=True, multichannel=True)

            # rank correlation between histogram of gradients
            normal_hog = feature.hog(normal_mask_0_1_norm,
                                     pixels_per_cell=(16, 16))
            rand_hog = feature.hog(rand_mask_0_1_norm,
                                   pixels_per_cell=(16, 16))
            rank_corr_hog = spr(normal_hog, rand_hog)[0]

            # collate all the values into their respective dictionaries.
            rank_correlation_dictionary_abs_norm[layer][method].append(
                rk_abs_abs_value_norm)
            rank_correlation_dictionary_no_abs_norm[layer][method].append(
                rk_no_abs_value_norm)
            ssim_dictionary_0_1_norm[layer][method].append(ss1)
            hog_dictionary_0_1_norm[layer][method].append(rank_corr_hog)

rk_mean_dictionary = {}
rk_std_dictionary = {}

for key in rank_correlation_dictionary_abs_norm:
    rk_mean_dictionary[key] = {}
    rk_std_dictionary[key] = {}
    for key2 in rank_correlation_dictionary_abs_norm[key]:
        rk_mean_dictionary[key][key2] = np.mean(
            rank_correlation_dictionary_abs_norm[key][key2])
        rk_std_dictionary[key][key2] = np.std(
            rank_correlation_dictionary_abs_norm[key][key2])

rk_mean_dictionary_no_abs = {}
rk_std_dictionary_no_abs = {}

for key in rank_correlation_dictionary_no_abs_norm:
    rk_mean_dictionary_no_abs[key] = {}
    rk_std_dictionary_no_abs[key] = {}
    for key2 in rank_correlation_dictionary_no_abs_norm[key]:
        rk_mean_dictionary_no_abs[key][key2] = np.mean(
            rank_correlation_dictionary_no_abs_norm[key][key2])
        rk_std_dictionary_no_abs[key][key2] = np.std(
            rank_correlation_dictionary_no_abs_norm[key][key2])

ssim_mean_dictionary = {}
ssim_std_dictionary = {}

for key in ssim_dictionary_0_1_norm:
    ssim_mean_dictionary[key] = {}
    ssim_std_dictionary[key] = {}
    for key2 in ssim_dictionary_0_1_norm[key]:
        ssim_mean_dictionary[key][key2] = np.mean(
            ssim_dictionary_0_1_norm[key][key2])
        ssim_std_dictionary[key][key2] = np.std(
            ssim_dictionary_0_1_norm[key][key2])

hog_mean_dictionary = {}
hog_std_dictionary = {}

for key in hog_dictionary_0_1_norm:
    hog_mean_dictionary[key] = {}
    hog_std_dictionary[key] = {}
    for key2 in hog_dictionary_0_1_norm[key]:
        hog_mean_dictionary[key][key2] = np.mean(
            hog_dictionary_0_1_norm[key][key2])
        hog_std_dictionary[key][key2] = np.std(
            hog_dictionary_0_1_norm[key][key2])


rk_df = pd.DataFrame(rk_mean_dictionary)
rk_df2 = pd.DataFrame(rk_std_dictionary)

# include no randomization
rk_df["Original"] = [1.0]*rk_df.shape[0]
rk_df2["Original"] = [0.0]*rk_df2.shape[0]

# switch order of the columns
layer_order = ['Original']
for val in layer_randomization_order:
    layer_order.append(val)
to_reverse = False
if to_reverse:
    layer_order = list(reversed(layer_order))
rk_df = rk_df.reindex(columns=layer_order)
rk_df2 = rk_df2.reindex(columns=layer_order)


layer_randomization_order_plotting = ['Original'] + layer_randomization_order

new_layer_order = []
for val in layer_randomization_order_plotting:
    # if val != 'classifier' and len(val) > 9:
    #     module = val[:8]
    #     layer = val[9:]
    #     val = f'{module}\n{layer}'
    if val != 'classifier':
        val = val.split('/')[-1]
    # if len(val.split('_')) > 1:
    #     val = val.split('_')[1]
    new_layer_order.append(val)

rk_df.columns = new_layer_order
rk_df2.columns = new_layer_order

# Rank Correlation between positive only masks
figsize = (7.5, 5)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.2, hspace=0.1)

sns.set(style="ticks")
# fig = plt.figure(figsize=figsize)
x = [i+1 for i in range(len(new_layer_order))]

ax0 = plt.subplot(gs[0, 0])
ax0.plot(x, rk_df.loc['Gradient', :].values,
         'ro-', lw=1, markersize=2, linestyle='--', label='Gradient')
ax0.fill_between(x, rk_df.loc['Gradient', :].values-rk_df2.loc['Gradient', :].values,
                 rk_df.loc['Gradient', :].values +
                 rk_df2.loc['Gradient', :].values,
                 facecolor='r', alpha=0.05)

ax0.plot(x, rk_df.loc['Gradient-SG', :].values,
         'yo-', lw=1, markersize=2, linestyle='--', label='Gradient-SG')
ax0.fill_between(x, rk_df.loc['Gradient-SG', :].values-rk_df2.loc['Gradient-SG', :].values,
                 rk_df.loc['Gradient-SG', :].values +
                 rk_df2.loc['Gradient-SG', :].values,
                 facecolor='m', alpha=0.05)

ax0.plot(x, rk_df.loc['Guided\nBackProp', :].values,
         'go-', lw=1, markersize=2, linestyle='--', label='Guided BackProp')
ax0.fill_between(x, rk_df.loc['Guided\nBackProp', :].values-rk_df2.loc['Guided\nBackProp', :].values,
                 rk_df.loc['Guided\nBackProp', :].values +
                 rk_df2.loc['Guided\nBackProp', :].values,
                 facecolor='g', alpha=0.05)


ax0.plot(x, rk_df.loc['GradCAM', :].values, 'co-',
         lw=1, markersize=2, linestyle='--', label='GradCAM')
ax0.fill_between(x, rk_df.loc['GradCAM', :].values-rk_df2.loc['GradCAM', :].values,
                 rk_df.loc['GradCAM', :].values +
                 rk_df2.loc['GradCAM', :].values,
                 facecolor='c', alpha=0.05)


ax0.plot(x, rk_df.loc['GBP-GC', :].values,
         'bo-', lw=1, markersize=2, linestyle='--', label='Guided GradCAM')
ax0.fill_between(x, rk_df.loc['GBP-GC', :].values-rk_df2.loc['GBP-GC', :].values,
                 rk_df.loc['GBP-GC', :].values+rk_df2.loc['GBP-GC', :].values,
                 facecolor='b', alpha=0.05)


# ax0.plot(x, rk_df.loc['Integrated\nGradients', :].values,
#          'mo-', lw=1, markersize=2, linestyle='--', label='Integrated\nGradients')
# ax0.fill_between(x, rk_df.loc['Integrated\nGradients', :].values-rk_df2.loc['Integrated\nGradients', :].values,
#                  rk_df.loc['Integrated\nGradients', :].values +
#                  rk_df2.loc['Integrated\nGradients', :].values,
#                  facecolor='m', alpha=0.05)

# ax0.plot(x, rk_df.loc['Input-Grad', :].values,
#          'yo-', lw=1, markersize=2, linestyle='--', label='Gradient-Input')
# ax0.fill_between(x, rk_df.loc['Input-Grad', :].values-rk_df2.loc['Input-Grad', :].values,
#                  rk_df.loc['Input-Grad', :].values +
#                  rk_df2.loc['Input-Grad', :].values,
#                  facecolor='y', alpha=0.05)

# ax0.plot(x, rk_df.loc['Occlusion', :].values,
#          'y^--', lw=1, markersize=2, linestyle='--', label='Occlusion')
# ax0.fill_between(x, rk_df.loc['Occlusion', :].values-rk_df2.loc['Occlusion', :].values,
#                  rk_df.loc['Occlusion', :].values +
#                  rk_df2.loc['Occlusion', :].values,
#                  facecolor='y', alpha=0.05)

ax0.plot(x, rk_df.loc['DeepLift', :].values,
         'm^--', lw=1, markersize=2, linestyle='--', label='DeepLift')
ax0.fill_between(x, rk_df.loc['DeepLift', :].values-rk_df2.loc['DeepLift', :].values,
                 rk_df.loc['DeepLift', :].values +
                 rk_df2.loc['DeepLift', :].values,
                 facecolor='y', alpha=0.05)

ax0.plot(x, rk_df.loc['Explanation\nMap', :].values,
         'ko-', lw=1, markersize=2, linestyle='--', label='Explanation\nMap')
ax0.fill_between(x, rk_df.loc['Explanation\nMap', :].values-rk_df2.loc['Explanation\nMap', :].values,
                 rk_df.loc['Explanation\nMap', :].values +
                 rk_df2.loc['Explanation\nMap', :].values,
                 facecolor='k', alpha=0.05)

ax0.set_xticks([])
ax0.set_xlabel("")
# ax0.set_xticks(rotation=-45)
# ax0.set_xticklabels(new_layer_order, rotation=-45)
# ax0.set_yticks([])
ax0.set_ylabel("Rank Correlation", **txt_font_prop)
ax0.set_yticks(ax0.get_yticks())
yticklabels = [str(round(x, 2)) for x in ax0.get_yticks()]
ax0.set_yticklabels(yticklabels, **axis_font_prop)
ax0.axhline(y=0.0, color='r', linestyle='--')
ax0.axvline(x=2.0, ymin=0.0, ymax=1.0, color='k', linestyle='--')
# ax0.text(-0.001, 0, 'A', **marker_font_prop)
# ax0.set_ylim([-0.5, 1.1])
# # plt.xlim([0.5, 4.5])
# plt.title("Cascading Experiment (Abs Value)")
# plt.legend(loc=8, ncol=2, fontsize=7, frameon=False)
# ax0.tick_params(axis='x', which='both', top='off')

# plt.legend(loc='upper left', bbox_to_anchor=(0.16, 0.2), ncol=3, frameon=False)
# plt.legend(loc='lower center', ncol=3, frameon=False)
# legend = plt.legend(loc='best', ncol=3, frameon=False)
# plt.show()
# fig.tight_layout()
# plt.savefig(saved_filepath_model / f'rank_correlation_pos.{format_}', format=format_, dpi=300)


# Rank Correlation between signed masks

rk_df_no_abs = pd.DataFrame(rk_mean_dictionary_no_abs)
rk_df2_no_abs = pd.DataFrame(rk_std_dictionary_no_abs)

# include no randomization
rk_df_no_abs["Original"] = [1.0]*rk_df_no_abs.shape[0]
rk_df2_no_abs["Original"] = [0.0]*rk_df2_no_abs.shape[0]

# switch order of the columns
# switch order of the columns
layer_order = ["Original"]
for val in layer_randomization_order:
    layer_order.append(val)
to_reverse = False  # reverse the axis if desired
if to_reverse:
    layer_order = list(reversed(layer_order))
rk_df_no_abs = rk_df_no_abs.reindex(columns=layer_order)
rk_df2_no_abs = rk_df2_no_abs.reindex(columns=layer_order)

rk_df_no_abs.columns = new_layer_order
rk_df2_no_abs.columns = new_layer_order

sns.set(style="ticks")
# fig = plt.figure(figsize=figsize)
x = [i+1 for i in range(len(new_layer_order))]

ax1 = plt.subplot(gs[0, 1])
ax1.plot(x, rk_df_no_abs.loc['Gradient', :].values,
         'ro-', lw=1, markersize=2, linestyle='--', label='Gradient')
ax1.fill_between(x, rk_df_no_abs.loc['Gradient', :].values-rk_df2_no_abs.loc['Gradient', :].values,
                 rk_df_no_abs.loc['Gradient', :].values +
                 rk_df2_no_abs.loc['Gradient', :].values,
                 facecolor='r', alpha=0.05)

ax1.plot(x, rk_df_no_abs.loc['Gradient-SG', :].values,
         'yo-', lw=1, markersize=2, linestyle='--', label='Gradient-SG')
ax1.fill_between(x, rk_df_no_abs.loc['Gradient-SG', :].values-rk_df2_no_abs.loc['Gradient-SG', :].values,
                 rk_df_no_abs.loc['Gradient-SG', :].values +
                 rk_df2_no_abs.loc['Gradient-SG', :].values,
                 facecolor='m', alpha=0.05)


ax1.plot(x, rk_df.loc['Guided\nBackProp', :].values,
         'go-', lw=1, markersize=2, linestyle='--', label='Guided BackProp')
ax1.fill_between(x, rk_df_no_abs.loc['Guided\nBackProp', :].values-rk_df2_no_abs.loc['Guided\nBackProp', :].values,
                 rk_df_no_abs.loc['Guided\nBackProp', :].values +
                 rk_df2_no_abs.loc['Guided\nBackProp', :].values,
                 facecolor='g', alpha=0.05)


ax1.plot(x, rk_df_no_abs.loc['GradCAM', :].values,
         'co-', lw=1, markersize=2, linestyle='--', label='GradCAM')
ax1.fill_between(x, rk_df_no_abs.loc['GradCAM', :].values-rk_df2_no_abs.loc['GradCAM', :].values,
                 rk_df_no_abs.loc['GradCAM', :].values +
                 rk_df2_no_abs.loc['GradCAM', :].values,
                 facecolor='c', alpha=0.05)


ax1.plot(x, rk_df_no_abs.loc['GBP-GC', :].values,
         'bo-', lw=1, markersize=2, linestyle='--', label='Guided GradCAM')
ax1.fill_between(x, rk_df_no_abs.loc['GBP-GC', :].values-rk_df2_no_abs.loc['GBP-GC', :].values,
                 rk_df_no_abs.loc['GBP-GC', :].values +
                 rk_df2_no_abs.loc['GBP-GC', :].values,
                 facecolor='b', alpha=0.05)


# ax1.plot(x, rk_df_no_abs.loc['Integrated\nGradients', :].values,
#          'mo-', lw=1, markersize=2, linestyle='--', label='Integrated\nGradients')
# ax1.fill_between(x, rk_df_no_abs.loc['Integrated\nGradients', :].values-rk_df2_no_abs.loc['Integrated\nGradients', :].values,
#                  rk_df_no_abs.loc['Integrated\nGradients', :].values +
#                  rk_df2_no_abs.loc['Integrated\nGradients', :].values,
#                  facecolor='m', alpha=0.05)

# ax1.plot(x, rk_df_no_abs.loc['Input-Grad', :].values,
#          'yo-', lw=1, markersize=2, linestyle='--', label='Gradient-Input')
# ax1.fill_between(x, rk_df_no_abs.loc['Input-Grad', :].values-rk_df2_no_abs.loc['Input-Grad', :].values,
#                  rk_df_no_abs.loc['Input-Grad', :].values +
#                  rk_df2_no_abs.loc['Input-Grad', :].values,
#                  facecolor='y', alpha=0.05)

# ax1.plot(x, rk_df.loc['Occlusion', :].values,
#          'y^--', lw=1, markersize=2, linestyle='--', label='Occlusion')
# ax1.fill_between(x, rk_df.loc['Occlusion', :].values-rk_df2.loc['Occlusion', :].values,
#                  rk_df.loc['Occlusion', :].values +
#                  rk_df2.loc['Occlusion', :].values,
#                  facecolor='y', alpha=0.05)

ax1.plot(x, rk_df.loc['DeepLift', :].values,
         'm^--', lw=1, markersize=2, linestyle='--', label='DeepLift')
ax1.fill_between(x, rk_df.loc['DeepLift', :].values-rk_df2.loc['DeepLift', :].values,
                 rk_df.loc['DeepLift', :].values +
                 rk_df2.loc['DeepLift', :].values,
                 facecolor='y', alpha=0.05)

ax1.plot(x, rk_df.loc['Explanation\nMap', :].values,
         'ko-', lw=1, markersize=2, linestyle='--', label='Explanation\nMap')
ax1.fill_between(x, rk_df.loc['Explanation\nMap', :].values-rk_df2.loc['Explanation\nMap', :].values,
                 rk_df.loc['Explanation\nMap', :].values +
                 rk_df2.loc['Explanation\nMap', :].values,
                 facecolor='k', alpha=0.05)

# ax1.set_xticks(x, new_layer_order)
ax1.set_xticks([])
ax1.set_xlabel("")
ax1.set_yticks(ax1.get_yticks())
yticklabels = [str(round(x, 2)) for x in ax1.get_yticks()]
ax1.set_yticklabels(yticklabels, **axis_font_prop)
# ax1.set_yticks([])
# ax1.set_ylabel("")
# ax1.set_ylabel("Rank Correlation")
ax1.axhline(y=0.0, color='r', linestyle='--')
ax1.axvline(x=2.0, ymin=0.0, ymax=1.0, color='k', linestyle='--')
# ax1.text(-0.001, 0, 'B')
# ax1.set_ylim([-0.2, 1.0])
# ax1.set_ylim([-0.6, 1.1])
# ax1.xlim([0.5, 4.5])
# ax1.title("Cascading Experiment (No Abs Value)")
# ax1.legend(loc=8, ncol=2, fontsize=7, frameon=False)
# ax1.tick_params(axis='x', which='both', top='off')

# ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -1.0), ncol=3)
# legend = plt.legend(loc='best', ncol=4, frameon=False)
# ax1.show()
# fig.tight_layout()
# plt.savefig(saved_filepath_model / f'rank_correlation_signed.{format_}', format=format_, dpi=300)


# SSIM Metric
ssim = pd.DataFrame(ssim_mean_dictionary)
ssim2 = pd.DataFrame(ssim_std_dictionary)

# include no randomization
ssim["Original"] = [1.0]*ssim.shape[0]
ssim2["Original"] = [0.0]*ssim2.shape[0]

# switch order of the columns
# switch order of the columns
layer_order = ["Original"]
for val in layer_randomization_order:
    layer_order.append(val)
to_reverse = False  # reverse the axis if desired
if to_reverse:
    layer_order = list(reversed(layer_order))
ssim = ssim.reindex(columns=layer_order)
ssim2 = ssim2.reindex(columns=layer_order)

ssim.columns = new_layer_order
ssim2.columns = new_layer_order

sns.set(style="ticks")
# fig = plt.figure(figsize=figsize)
ax2 = plt.subplot(gs[1, 0])
x = [i+1 for i in range(len(new_layer_order))]

ax2.plot(x, ssim.loc['Gradient', :].values,
         'ro-', lw=1, markersize=2, linestyle='--', label='Gradient')
ax2.fill_between(x, ssim.loc['Gradient', :].values-ssim2.loc['Gradient', :].values,
                 ssim.loc['Gradient', :].values +
                 ssim2.loc['Gradient', :].values,
                 facecolor='r', alpha=0.05)

ax2.plot(x, ssim.loc['Gradient-SG', :].values,
         'yo-', lw=1, markersize=2, linestyle='--', label='Gradient-SG')
ax2.fill_between(x, ssim.loc['Gradient-SG', :].values-ssim2.loc['Gradient-SG', :].values,
                 ssim.loc['Gradient-SG', :].values +
                 ssim2.loc['Gradient-SG', :].values,
                 facecolor='m', alpha=0.05)

ax2.plot(x, ssim.loc['Guided\nBackProp', :].values,
         'go-', lw=1, markersize=2, linestyle='--', label='Guided BackProp')
ax2.fill_between(x, ssim.loc['Guided\nBackProp', :].values-ssim2.loc['Guided\nBackProp', :].values,
                 ssim.loc['Guided\nBackProp', :].values +
                 ssim2.loc['Guided\nBackProp', :].values,
                 facecolor='g', alpha=0.05)


ax2.plot(x, ssim.loc['GradCAM', :].values, 'co-',
         lw=1, markersize=2, linestyle='--', label='GradCAM')
ax2.fill_between(x, ssim.loc['GradCAM', :].values-ssim2.loc['GradCAM', :].values,
                 ssim.loc['GradCAM', :].values+ssim2.loc['GradCAM', :].values,
                 facecolor='c', alpha=0.05)


ax2.plot(x, ssim.loc['GBP-GC', :].values,
         'bo-', lw=1, markersize=2, linestyle='--', label='Guided GradCAM')
ax2.fill_between(x, ssim.loc['GBP-GC', :].values-ssim2.loc['GBP-GC', :].values,
                 ssim.loc['GBP-GC', :].values+ssim2.loc['GBP-GC', :].values,
                 facecolor='b', alpha=0.05)


# ax2.plot(x, ssim.loc['Integrated\nGradients', :].values,
#          'mo-', lw=1, markersize=2, linestyle='--', label='Integrated\nGradients')
# ax2.fill_between(x, ssim.loc['Integrated\nGradients', :].values-ssim2.loc['Integrated\nGradients', :].values,
#                  ssim.loc['Integrated\nGradients', :].values +
#                  ssim2.loc['Integrated\nGradients', :].values,
#                  facecolor='m', alpha=0.05)

# ax2.plot(x, ssim.loc['Input-Grad', :].values,
#          'yo-', lw=1, markersize=2, linestyle='--', label='Gradient-Input')
# ax2.fill_between(x, ssim.loc['Input-Grad', :].values-ssim2.loc['Input-Grad', :].values,
#                  ssim.loc['Input-Grad', :].values +
#                  ssim2.loc['Input-Grad', :].values,
#                  facecolor='y', alpha=0.05)

# ax2.plot(x, rk_df.loc['Occlusion', :].values,
#          'y^--', lw=1, markersize=2, linestyle='--', label='Occlusion')
# ax2.fill_between(x, rk_df.loc['Occlusion', :].values-rk_df2.loc['Occlusion', :].values,
#                  rk_df.loc['Occlusion', :].values +
#                  rk_df2.loc['Occlusion', :].values,
#                  facecolor='y', alpha=0.05)

ax2.plot(x, rk_df.loc['DeepLift', :].values,
         'm^--', lw=1, markersize=2, linestyle='--', label='DeepLift')
ax2.fill_between(x, rk_df.loc['DeepLift', :].values-rk_df2.loc['DeepLift', :].values,
                 rk_df.loc['DeepLift', :].values +
                 rk_df2.loc['DeepLift', :].values,
                 facecolor='y', alpha=0.05)

ax2.plot(x, rk_df.loc['Explanation\nMap', :].values,
         'ko-', lw=1, markersize=2, linestyle='--', label='Explanation\nMap')
ax2.fill_between(x, rk_df.loc['Explanation\nMap', :].values-rk_df2.loc['Explanation\nMap', :].values,
                 rk_df.loc['Explanation\nMap', :].values +
                 rk_df2.loc['Explanation\nMap', :].values,
                 facecolor='k', alpha=0.05)

ax2.set_xticks(x)
# ax2.set_xlabel("")
# ax2.set_xticks(rotation=-45)

if opt.model_type == 'VGG':
    new_layer_order = ['Original', 'Classifier', 'Conv13', 'Conv12', 'Conv10', 'Conv7', 'Conv1']
elif opt.model_type == 'ResNet':
    new_layer_order = ['Original', 'FC', 'Block4', 'Block3', 'Block2', 'Block1', 'Conv1']
else:
    new_layer_order = ['Original', 'FC', 'Mixed_7c', 'Mixed_7b', 'Mixed_6e', 'Mixed_5b', 'Conv_1a']

ax2.set_xticklabels(new_layer_order, rotation=-45, **axis_label_font_prop)
ax2.set_ylabel("Correlation", **txt_font_prop)
ax2.set_yticks(ax2.get_yticks())
yticklabels = [str(round(x, 2)) for x in ax2.get_yticks()]
ax2.set_yticklabels(yticklabels, **axis_font_prop)
ax2.axhline(y=0.0, color='r', linestyle='--')
ax2.axvline(x=2.0, ymin=0.0, ymax=1.0, color='k', linestyle='--')
# ax2.text(-0.2, 0, 'C')
# ax2.set_ylim([-0.4, 1.1])
# ax2.ylim([0.0, 1.2])
# ax2.xlim([0.5, 4.5])
# ax2.title("SSIM")
# ax2.legend(loc=8, ncol=2, fontsize=7, frameon=False)
# ax2.tick_params(axis='x', which='both', top='off')

# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -1.0), ncol=3)
# legend = plt.legend(loc='best', ncol=4, frameon=False)
# plt.show()
# fig.tight_layout()
# plt.savefig(saved_filepath_model / f'ssim.{format_}', format=format_, dpi=300)


# Rank Correlation between Histogram of Gradients
rkhog = pd.DataFrame(hog_mean_dictionary)
rkhog2 = pd.DataFrame(hog_std_dictionary)

# include no randomization
rkhog["Original"] = [1.0]*rk_df.shape[0]
rkhog2["Original"] = [0.0]*rk_df2.shape[0]

# switch order of the columns
layer_order = ["Original"]
for val in layer_randomization_order:
    layer_order.append(val)
to_reverse = False  # reverse the axis if desired
if to_reverse:
    layer_order = list(reversed(layer_order))
rkhog = rkhog.reindex(columns=layer_order)
rkhog2 = rkhog2.reindex(columns=layer_order)

rkhog.columns = new_layer_order
rkhog2.columns = new_layer_order

sns.set(style="ticks")
# fig = plt.figure(figsize=figsize)
ax3 = plt.subplot(gs[1, 1])
x = [i+1 for i in range(len(new_layer_order))]

ax3.plot(x, rkhog.loc['Gradient', :].values,
         'ro-', lw=1, markersize=2, linestyle='--', label='Gradient')
ax3.fill_between(x, rkhog.loc['Gradient', :].values-rkhog2.loc['Gradient', :].values,
                 rkhog.loc['Gradient', :].values +
                 rkhog2.loc['Gradient', :].values,
                 facecolor='r', alpha=0.05)

ax3.plot(x, rkhog.loc['Gradient-SG', :].values,
         'yo-', lw=1, markersize=2, linestyle='--', label='Gradient-SG')
ax3.fill_between(x, rkhog.loc['Gradient-SG', :].values-rkhog2.loc['Gradient-SG', :].values,
                 rkhog.loc['Gradient-SG', :].values +
                 rkhog2.loc['Gradient-SG', :].values,
                 facecolor='m', alpha=0.05)

ax3.plot(x, rkhog.loc['Guided\nBackProp', :].values,
         'go-', lw=1, markersize=2, linestyle='--', label='Guided BackProp')
ax3.fill_between(x, rkhog.loc['Guided\nBackProp', :].values-rkhog2.loc['Guided\nBackProp', :].values,
                 rkhog.loc['Guided\nBackProp', :].values +
                 rkhog2.loc['Guided\nBackProp', :].values,
                 facecolor='g', alpha=0.05)


ax3.plot(x, rkhog.loc['GradCAM', :].values, 'co-',
         lw=1, markersize=2, linestyle='--', label='GradCAM')
ax3.fill_between(x, rkhog.loc['GradCAM', :].values-rkhog2.loc['GradCAM', :].values,
                 rkhog.loc['GradCAM', :].values +
                 rkhog2.loc['GradCAM', :].values,
                 facecolor='c', alpha=0.05)


ax3.plot(x, rkhog.loc['GBP-GC', :].values,
         'bo-', lw=1, markersize=2, linestyle='--', label='Guided GradCAM')
ax3.fill_between(x, rkhog.loc['GBP-GC', :].values-rkhog2.loc['GBP-GC', :].values,
                 rkhog.loc['GBP-GC', :].values+rkhog2.loc['GBP-GC', :].values,
                 facecolor='b', alpha=0.05)


# ax3.plot(x, rkhog.loc['Integrated\nGradients', :].values,
#          'mo-', lw=1, markersize=2, linestyle='--', label='Integrated\nGradients')
# ax3.fill_between(x, rkhog.loc['Integrated\nGradients', :].values-rkhog2.loc['Integrated\nGradients', :].values,
#                  rkhog.loc['Integrated\nGradients', :].values +
#                  rkhog2.loc['Integrated\nGradients', :].values,
#                  facecolor='m', alpha=0.05)

# ax3.plot(x, rkhog.loc['Input-Grad', :].values,
#          'yo-', lw=1, markersize=2, linestyle='--', label='Gradient-Input')
# ax3.fill_between(x, rkhog.loc['Input-Grad', :].values-rkhog2.loc['Input-Grad', :].values,
#                  rkhog.loc['Input-Grad', :].values +
#                  rkhog2.loc['Input-Grad', :].values,
#                  facecolor='y', alpha=0.05)

# ax3.plot(x, rk_df.loc['Occlusion', :].values,
#          'y^--', lw=1, markersize=2, linestyle='--', label='Occlusion')
# ax3.fill_between(x, rk_df.loc['Occlusion', :].values-rk_df2.loc['Occlusion', :].values,
#                  rk_df.loc['Occlusion', :].values +
#                  rk_df2.loc['Occlusion', :].values,
#                  facecolor='y', alpha=0.05)

ax3.plot(x, rk_df.loc['DeepLift', :].values,
         'm^--', lw=1, markersize=2, linestyle='--', label='DeepLift')
ax3.fill_between(x, rk_df.loc['DeepLift', :].values-rk_df2.loc['DeepLift', :].values,
                 rk_df.loc['DeepLift', :].values +
                 rk_df2.loc['DeepLift', :].values,
                 facecolor='y', alpha=0.05)

ax3.plot(x, rk_df.loc['Explanation\nMap', :].values,
         'ko-', lw=1, markersize=2, linestyle='--', label='Explanation\nMap')
ax3.fill_between(x, rk_df.loc['Explanation\nMap', :].values-rk_df2.loc['Explanation\nMap', :].values,
                 rk_df.loc['Explanation\nMap', :].values +
                 rk_df2.loc['Explanation\nMap', :].values,
                 facecolor='k', alpha=0.05)

# ax3.set_xticks(x)
ax3.set_xticks([])
ax3.set_xlabel("")
# ax3.set_yticks([])
# ax3.set_ylabel("")
ax3.set_yticks(ax3.get_yticks())
yticklabels = [str(round(x, 2)) for x in ax3.get_yticks()]
ax3.set_yticklabels(yticklabels, **axis_font_prop)
# ax3.set_xticks(rotation=-45)
ax3.axhline(y=0.0, color='r', linestyle='--')
ax3.axvline(x=2.0, ymin=0.0, ymax=1.0, color='k', linestyle='--')
# ax3.text(-0.2, 0, 'D')
# ax3.set_ylim([0.0, 1.0])
# ax3.xlim([0.5, 4.5])
# ax3.set_ylabel("Rank Correlation of HOG")
# ax3.title("HOG Metric")
# ax3.legend(loc='upper right', ncol=1, fontsize=6, frameon=False)
ax3.tick_params(axis='x', which='both', top='off')

# Legend
handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc='upper center', fontsize=6, frameon=False)
plt.yticks(fontsize=8)

# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -1.0), ncol=3)
# legend = plt.legend(loc='best', ncol=4, frameon=False)
# plt.show()
# plt.savefig(saved_filepath_model / f'rank_correlation_hist_gradient.{format_}', format=format_, dpi=300)

plt.savefig(saved_filepath_model /
            f'correlation.{format_}', format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

# Single figure
fig, axs = plt.subplots(figsize=(3, 2))

axs.plot(x, rkhog.loc['Gradient', :].values,
         'ro-', lw=1, markersize=2, linestyle='--', label='Gradient')
axs.fill_between(x, rkhog.loc['Gradient', :].values-rkhog2.loc['Gradient', :].values,
                 rkhog.loc['Gradient', :].values +
                 rkhog2.loc['Gradient', :].values,
                 facecolor='r', alpha=0.05)

axs.plot(x, rkhog.loc['Gradient-SG', :].values,
         'yo-', lw=1, markersize=2, linestyle='--', label='Gradient-SG')
axs.fill_between(x, rkhog.loc['Gradient-SG', :].values-rkhog2.loc['Gradient-SG', :].values,
                 rkhog.loc['Gradient-SG', :].values +
                 rkhog2.loc['Gradient-SG', :].values,
                 facecolor='m', alpha=0.05)

axs.plot(x, rkhog.loc['GradCAM', :].values, 'co-',
         lw=1, markersize=2, linestyle='--', label='GradCAM')
axs.fill_between(x, rkhog.loc['GradCAM', :].values-rkhog2.loc['GradCAM', :].values,
                 rkhog.loc['GradCAM', :].values +
                 rkhog2.loc['GradCAM', :].values,
                 facecolor='c', alpha=0.05)

axs.plot(x, rk_df.loc['DeepLift', :].values,
         'm^--', lw=1, markersize=2, linestyle='--', label='DeepLift')
axs.fill_between(x, rk_df.loc['DeepLift', :].values-rk_df2.loc['DeepLift', :].values,
                 rk_df.loc['DeepLift', :].values +
                 rk_df2.loc['DeepLift', :].values,
                 facecolor='y', alpha=0.05)

axs.plot(x, rk_df.loc['Explanation\nMap', :].values,
         'ko-', lw=1, markersize=2, linestyle='--', label='Explanation\nMap')
axs.fill_between(x, rk_df.loc['Explanation\nMap', :].values-rk_df2.loc['Explanation\nMap', :].values,
                 rk_df.loc['Explanation\nMap', :].values +
                 rk_df2.loc['Explanation\nMap', :].values,
                 facecolor='k', alpha=0.05)

axs.plot(x, rkhog.loc['Guided\nBackProp', :].values,
         'go-', lw=1, markersize=2, linestyle='--', label='Guided BackProp')
axs.fill_between(x, rkhog.loc['Guided\nBackProp', :].values-rkhog2.loc['Guided\nBackProp', :].values,
                 rkhog.loc['Guided\nBackProp', :].values +
                 rkhog2.loc['Guided\nBackProp', :].values,
                 facecolor='g', alpha=0.05)

axs.plot(x, rkhog.loc['GBP-GC', :].values,
         'bo-', lw=1, markersize=2, linestyle='--', label='Guided GradCAM')
axs.fill_between(x, rkhog.loc['GBP-GC', :].values-rkhog2.loc['GBP-GC', :].values,
                 rkhog.loc['GBP-GC', :].values+rkhog2.loc['GBP-GC', :].values,
                 facecolor='b', alpha=0.05)


if opt.model_type == 'VGG':
    new_layer_order = ['Original', 'Classifier', 'Conv13', 'Conv12', 'Conv10', 'Conv7', 'Conv1']
elif opt.model_type == 'ResNet':
    new_layer_order = ['Original', 'FC', 'Block4', 'Block3', 'Block2', 'Block1', 'Conv1']
else:
    new_layer_order = ['Original', 'FC', 'Mixed_7c', 'Mixed_7b', 'Mixed_6e', 'Mixed_5b', 'Conv_1a']

# axs.set_xticks(x, new_layer_order, rotation=-45, **axis_label_font_prop)
axs.set_xticks(x)
axs.set_xticklabels(new_layer_order, rotation=-45, **axis_label_font_prop)
axs.set_yticks([-0.5, 0, 0.5, 1.0])
if opt.model_type == 'VGG':
    axs.set_yticklabels([-0.5, 0, 0.5, 1.0], **axis_font_prop)
    axs.set_ylabel('Similarity', **axis_label_font_prop)
else:
    axs.set_yticklabels([])
    axs.set_ylabel('')

axs.axhline(y=0.0, color='r', linestyle='--')
axs.axvline(x=2.0, ymin=0.0, ymax=1.0, color='k', linestyle='--')

if opt.model_type == 'Inception3':
    axs.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=6, frameon=False)
# plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
plt.savefig(saved_filepath_model /
            f'ax3_figure.{format_}', format=format_, dpi=300, bbox_inches='tight', pad_inches=0)

# # Save just the portion _inside_ the second axis's boundaries
# extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# # fig.savefig(f'ax3_figure.{format_}', format=format_, dpi=300, bbox_inches=extent)

# # Pad the saved area by 10% in the x-direction and 20% in the y-direction
# fig.savefig(saved_filepath_model /
#             f'ax3_figure_expanded.{format_}', format=format_, dpi=300, bbox_inches=extent.expanded(1.2, 1.2))