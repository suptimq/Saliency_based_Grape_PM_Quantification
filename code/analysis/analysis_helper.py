import math
import json
import numpy as np
import cv2


import torch
import torch.nn as nn
from torchvision import models


def slding_window(length, fix, step_size, ceil=False):
    """
        Calculate the total number of steps
    Args:
        length:        Total length
        fix:           Fixed length like kernel size in the convolution op
        step_size:     Step length like stride in the convolution op
    """
    steps = 0
    if ceil:
        steps = math.ceil((length - fix) / step_size) + 1
    else:
        steps = math.floor((length - fix) / step_size) + 1

    return steps


def pad_images(img, step_size, IMG_WIDTH, IMG_HEIGHT):
    """
        Add padding into images
    """
    width, height = img.size
    subim_x = slding_window(width, IMG_WIDTH, step_size, ceil=True)
    subim_y = slding_window(height, IMG_HEIGHT, step_size, ceil=True)

    # Calculate padding for the right and bottom side
    padding_x = (subim_x - 1) * step_size + IMG_WIDTH - width
    padding_y = (subim_y - 1) * step_size + IMG_HEIGHT - height

    return [padding_x, padding_y]


def resize_images(img, padding):
    """
    Return: 
        Resized images
    """
    padding_x, padding_y = padding
    width, height = img.size
    shape = (width + padding_x, height + padding_y)

    return img.resize(shape)
