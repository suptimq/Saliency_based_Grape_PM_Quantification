import os
import numpy as np

from matplotlib import gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from visualization.viz_util import (
    visualize_image_attr, visualize_image_attr_multiple, abs_grayscale_norm, diverging_norm, normalize_image)


def get_last_conv_layer(model):
    """
        Get the last convolutional layer given model
    """
    model_name = model.__class__.__name__
    if model_name == 'VGG':
        last_conv_layer = model.features[28]
    elif model_name == 'SqueezeNet':
        last_conv_layer = model.features[12].expand3x3
    elif model_name == 'ResNet':
        last_conv_layer = model.layer4[-1].conv2
    elif model_name == 'Inception3':
        last_conv_layer = model.Mixed_7c

    return last_conv_layer


def get_first_conv_layer(model):
    """
        Get the first convolution layer given model
    """
    model_name = model.__class__.__name__
    if model_name == 'VGG':
        return 'features.0'
    elif model_name == 'ResNet':
        return 'conv1'
    elif model_name == 'Inception3':
        return 'Conv2d_1a_3x3'


def viz_image_attr(original_image, image_attr_dict, lost_focus_pixel_index, cmap, text='', signs=["positive", "all"], label=None, outlier_perc=5, figsize=(10, 10), alpha_overlay=0.4):
    """
        Generate visualization figures using various techniques
    Args:
        original_image:     Original images for the purpose of blending
        image_attr_dict:    Dictionary for attributions, where key is the technique and value is the outputted value array
        cmap:               Color map
    Return:
        heatmap_figs:               plt figure heatmap objects
        hist_figs:                  plt figure histogram objects
    """
    heatmap_figs = {}
    hist_figs = {}
    for key, value in image_attr_dict.items():
        if key == 'Guided-GradCam' or key == 'Explanation Map':
            value = value * 255
        if label:
            titles = [f'Blended {key} ({label})', f'{key} heatmap ({label})']
        else:
            titles = [f'Blended {key}', f'{key} heatmap']
        heatmap_fig, _, norm_attrs = visualize_image_attr_multiple(
            attr=value,
            original_image=original_image,
            alpha_overlay=alpha_overlay,
            methods=["blended_heat_map", "heat_map"],
            signs=signs,
            outlier_perc=outlier_perc,
            cmap=cmap,
            fig_size=figsize,
            titles=titles,
            use_pyplot=False,
            show_colorbar=True,
            text=text
        )
        heatmap_figs[key] = heatmap_fig

        hist_fig, hist_axis = plt.subplots(figsize=figsize)
        n1 = norm_attrs[1].reshape(1, -1).squeeze(0)
        on_foucs_pixel = np.delete(n1, lost_focus_pixel_index)
        hist_axis.hist(on_foucs_pixel)
        hist_figs[key] = hist_fig

    return heatmap_figs, hist_figs


def normalize_image_attr(original_image, image_attr_dict, **kwargs):
    """
        Doing normalization of the different attributions
    """
    # normalize by absolute values
    new_dict_abs_norm = {}
    # normalize but keep the signs
    new_dict_no_abs_norm = {}
    new_dict_0_1_norm = {}

    new_dict_abs_norm['Original'] = original_image
    new_dict_no_abs_norm['Original'] = original_image
    new_dict_0_1_norm['Original'] = original_image

    save_histogram = kwargs.get('hist', False)

    for key in image_attr_dict:
        mask = image_attr_dict[key]
        mask_abs_norm = abs_grayscale_norm(mask)
        mask_no_abs_norm = diverging_norm(mask)
        mask_0_1_norm = normalize_image(mask)

        if save_histogram:
            new_dict_abs_norm[key] = mask_abs_norm
            new_dict_abs_norm[key+'_hist'] = mask_abs_norm.reshape(1, -1)
            new_dict_no_abs_norm[key] = mask_no_abs_norm
            new_dict_no_abs_norm[key+'_hist'] = mask_no_abs_norm.reshape(1, -1)
            new_dict_0_1_norm[key] = mask_0_1_norm
            new_dict_0_1_norm[key+'_hist'] = mask_0_1_norm.reshape(1, -1)
        else:
            new_dict_abs_norm[key] = mask_abs_norm
            new_dict_no_abs_norm[key] = mask_no_abs_norm
            new_dict_0_1_norm[key] = mask_0_1_norm

    return new_dict_abs_norm, new_dict_no_abs_norm, new_dict_0_1_norm


def plot_figs(output_folder, imagename, figs, signs, figsize=(8, 5), default_cmap='bwr', **kwargs):
    assert len(figs) == len(signs)

    for fi in range(len(figs)):
        nfigs = len(figs[fi])
        nrows = 2
        ncols = 3

        finish = False
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows, ncols,
                               wspace=0.2, hspace=0.4)

        count = 0
        cmap = 'coolwarm' if signs[fi] == 'all' else default_cmap
        titles = list(figs[fi].keys())

        iou = kwargs.get('iou', [])
        l2 = kwargs.get('l2', [])
        gt_infected_pixel = kwargs.get('gt_infected_pixel', 0)

        for i in range(nrows):
            for j in range(ncols):
                fig_ = figs[fi]
                title = titles[count]
                ax = plt.subplot(gs[i, j])
                if 'hist' in title:
                    ax.hist(fig_[title].squeeze(0))
                    ax.set_xticks([0.2, 1])
                else:
                    heat_map = ax.imshow(fig_[title], vmin=0, vmax=1.0, cmap=cmap)
                    # ax.get_xaxis().set_visible(False)
                    # ax.get_yaxis().set_visible(False)
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    if count > 0 and count < nfigs - 1:
                        if iou and l2:
                            # ax.set_xlabel(f'IOU:{iou[count-1]}% Dice:{l2[count-1]}', fontsize=6)
                            ax.text(0.5, 1.1, f'IOU:{iou[count-1]}% Dice:{l2[count-1]}', fontsize=8)
                    if count == nfigs - 1:
                        if gt_infected_pixel:
                            ax.set_xlabel(f'number of infected pixels {gt_infected_pixel}')
                    
                ax.set_title(title, fontsize=12)

                count += 1
                if count == nfigs:
                    finish = True
                    break

            if finish:
                break

        label = kwargs.get('label', '')
        colorbar = kwargs.get('colorbar', False)
        if label:
            fig.suptitle(f'Prediction {label}')
        if colorbar:
            # axis_separator = make_axes_locatable(ax)
            # colorbar_axis = axis_separator.append_axes(
            #     "bottom", size="5%", pad=0.1)
            fig.subplots_adjust(bottom=0.2)
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
            if heat_map:
                # fig.colorbar(heat_map, orientation="horizontal",
                #              cax=colorbar_axis)
                fig.colorbar(heat_map, orientation="horizontal", cax=cbar_ax)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        plt.savefig(os.path.join(output_folder, f'{signs[fi]}_' + imagename), format='png', dpi=300)
        plt.close()
        print(f'Saved {signs[fi]}_{imagename}')


def save_figs(output_folder, imagename, figs, patch_idx=None):
    """
        Save visualized results to local
    Args:
        output_folder:    
        imagename:        Current image filename
        figs:             Dictionary for figures, where key is the technique and value is the outputted figure
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for key, value in figs.items():
        imagename_patch = imagename + (f'-{patch_idx}' if patch_idx else '')
        output_filepath = os.path.join(
            output_folder, f'{imagename_patch}-{key}.png')

        value.savefig(output_filepath)
        print(f'Saved {output_filepath}')
