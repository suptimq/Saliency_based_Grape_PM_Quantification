import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map

from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.morphology import medial_axis, skeletonize, binary_dilation


def hard_thresholding(mask, threshold, vmin=0, vmax=1):
    """
    Args:
        vmin:        Value represents healthy pixels
        vmax:        Value represents infected pixels
    """
    mask_copy = mask.copy()
    mask_copy[mask_copy < threshold] = vmin
    mask_copy[mask_copy >= threshold] = vmax

    return mask_copy.astype('uint8')


def otsu_thresholding(mask, vmin=0, vmax=1):
    mask_copy = (mask.copy() * 255).astype('uint8')
    _, th1 = cv2.threshold(mask_copy, vmin, vmax,
                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return th1


def skeletonization(mask):
    skeleton = skeletonize(mask)
    dilated_skeleton = binary_dilation(skeleton, selem=np.ones((5, 5)))

    return dilated_skeleton


def IoU_metric(saliency_mask, gt_mask, vmin=0, vmax=1):
    saliency_mask_copy = saliency_mask.copy()
    saliency_mask_copy[saliency_mask_copy == vmin] = 100

    gt_infected_pixel = len(gt_mask[gt_mask == vmax])
    mask_infected_pixel = len(saliency_mask_copy[saliency_mask_copy == vmax])
    overlapping = np.sum(saliency_mask_copy == gt_mask)

    iou = round(overlapping / (gt_infected_pixel +
                               mask_infected_pixel - overlapping) * 100, 2)

    return iou


def Dice_metric(saliency_mask, gt_mask, vmin=0, vmax=1):
    saliency_mask_copy = saliency_mask.copy()
    saliency_mask_copy[saliency_mask_copy == vmin] = 100

    gt_infected_pixel = len(gt_mask[gt_mask == vmax])
    mask_infected_pixel = len(saliency_mask_copy[saliency_mask_copy == vmax])
    overlapping = np.sum(saliency_mask_copy == gt_mask)

    dice = round(overlapping / (gt_infected_pixel +
                                mask_infected_pixel) * 2 * 100, 2)

    return dice


def L2_metric(saliency_mask, gt_mask, vmin=0):
    saliency_mask_copy = saliency_mask.copy()
    saliency_mask_copy[saliency_mask_copy == vmin] = 0

    return np.sum(saliency_mask_copy != gt_mask)


def apply_colormap_on_image(org_im, activation, colormap='Reds'):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = no_trans_heatmap.copy()
    heatmap[:, :, 3] = 0.3
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(
        heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)

    return no_trans_heatmap, heatmap_on_image


def plot_colormap_on_image(org_im, activation, colormap='Reds', colorbar=True, figsize=(6, 6)):
    """
        Plot heatmap on image by pyplot
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap (str): Name of the colormap
    """
    plt_fig, plt_axis = plt.subplots(figsize=(6, 6))
    # plt_axis.imshow(
    #     np.mean(org_im, axis=2), cmap='gray')
    plt_axis.imshow(org_im)
    heat_map = plt_axis.imshow(
        activation, cmap=colormap, vmin=0, vmax=1, alpha=0.3)
    if colorbar:
        # Colorbar
        axis_separator = make_axes_locatable(plt_axis)
        colorbar_axis = axis_separator.append_axes(
            "bottom", size="5%", pad=0.3)
        plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)

    return plt_fig, plt_axis
