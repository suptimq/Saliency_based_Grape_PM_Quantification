import sys
import numpy as np

sys.path.append('..')

from utils import hard_thresholding, otsu_thresholding

from analyzer_config import IMG_HEIGHT, IMG_WIDTH


"""
Deprecated
"""


def metric(patch_info, heatmap_info, threshold_info):
    """
        Calculate pixel level (mask out uncertain pixels) severity rate using gradcam heatmap
    Args:
        patch_info:         patch related information
        heatmap_info:       heatmap related information
        threshold_info:     threshold related information
    """

    patch_down_th = threshold_info['patch_down_th']
    patch_up_th = threshold_info['patch_up_th']
    pixel_th = threshold_info['pixel_th']

    prob_heatmap = heatmap_info.pop('prob_heatmap', None)
    mask_heatmap = prob_heatmap.copy()
    mask_heatmap[mask_heatmap == -np.inf] = 0.0      # lost focused pixels
    mask_heatmap[(mask_heatmap > patch_down_th) & (
        mask_heatmap < patch_up_th)] = 0.0           # uncertain pixels
    mask_heatmap = (mask_heatmap != 0.0)

    lost_focus_patch = patch_info['lost_focus_patch']
    lost_focus_pixel = lost_focus_patch * IMG_HEIGHT * IMG_WIDTH

    severity_rates = {}
    infected_pixels = {}
    total_pixels = {}

    for th in pixel_th:
        for key, val in heatmap_info.items():
            if th == 'otsu':
                saliency_mask = otsu_thresholding(val, vmin=0, vmax=1)
            else:
                saliency_mask = hard_thresholding(val, float(th), vmin=0, vmax=1)
            saliency_mask = saliency_mask * mask_heatmap

            total_pixel = saliency_mask.shape[0] * saliency_mask.shape[1]
            infected_pixel = len(saliency_mask[saliency_mask == 1])
            clear_pixel = len(saliency_mask[saliency_mask == 0])
            assert clear_pixel + infected_pixel == total_pixel

            if not severity_rates.get(th, None):
                severity_rates[th] = {}
            severity_rates[th][key] = round(
                infected_pixel / (total_pixel - lost_focus_pixel) * 100, 2)

            if not infected_pixels.get(th, None):
                infected_pixels[th] = {}
            infected_pixels[th][key] = infected_pixel
            if not total_pixels.get(th, None):
                total_pixels[th] = {}
            total_pixels[th][key] = total_pixel - lost_focus_pixel

    return severity_rates, (infected_pixels, total_pixels)
