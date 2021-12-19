import sys
import numpy as np

sys.path.append('..')

from analyzer_config import IMG_HEIGHT, IMG_WIDTH

def metric(patch_info, heatmap_info, threshold_info):
    """
        Calculate patch level severity rate using probability heatmap
    """
    prob_heatmap = heatmap_info['prob_heatmap']
    patch_down_th = threshold_info['patch_down_th']
    patch_up_th = threshold_info['patch_up_th']

    infected_patch = patch_info['infected_patch']
    clear_patch = patch_info['clear_patch']

    infected_pixel = len(prob_heatmap[prob_heatmap >= patch_up_th])
    clear_pixel = len(prob_heatmap[prob_heatmap <= patch_down_th]
                      ) - len(prob_heatmap[prob_heatmap == -np.inf])

    # assert infected_patch * IMG_HEIGHT * IMG_WIDTH == infected_pixel, 'error'
    # assert clear_patch * IMG_HEIGHT * IMG_WIDTH == clear_pixel, 'error'

    return round((infected_patch / (infected_patch + clear_patch)) * 100, 2), (infected_pixel, clear_pixel)
    # return round((infected_pixel / (infected_pixel + clear_pixel) * 100), 2)
