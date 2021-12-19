import cv2
import numpy as np

import pdb


def leaf_mask(img, rel_th=0.1):
    """
        Fill holes in a binary image using cv2
        # Relative thresholding (0 < rth < 1)
        # PMbot works best with 0.2
        # Blackbird works best with 0.1
    Args:
        img:        A RGB format image with PIL format
        rel_th:     Relative threshold value
    Return:
        im_out:     Foreground mask with Numpy array format
    """
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    wsize = 50
    wstep = 6

    res_offset = int(wsize / wstep)
    # center_offset = wsize / 2

    height, width, _ = img.shape
    aux_height = int((height / wstep) - res_offset)
    aux_width = int((width / wstep) - res_offset)

    # Output focal metric image
    fmat = np.ndarray(shape=(aux_height, aux_width), dtype=np.float64)

    coor_x = coor_y = 0

    # Get sharpness/focused of the image by calculating the standard deviation
    # Using sliding window
    for row in range(aux_height):
        for col in range(aux_width):
            kernel = img[coor_y: coor_y + wsize, coor_x: coor_x + wsize, :]
            _, stddev = cv2.meanStdDev(kernel)
            maxfm = stddev[0]
            for i in range(1, 3):
                maxfm = max(maxfm, stddev[i])
            fmat[row, col] = maxfm
            coor_x = coor_x + wstep
        coor_x = 0
        coor_y = coor_y + wstep

    _, max_val, _, _ = cv2.minMaxLoc(fmat)
    # 1st Threshold
    _, imbin = cv2.threshold(fmat, max_val * rel_th,
                             255, cv2.THRESH_BINARY_INV)
    imbin = imbin.astype('uint8')
    # Calculate the mean between mask and thresholded mask
    fv = cv2.mean(fmat, imbin)
    th = fv[0]
    # 2nd Threshold
    _, imbin = cv2.threshold(fmat, th, 255, cv2.THRESH_BINARY_INV)
    imbin = imbin.astype('uint8')
    # Bitwise not
    imbin1 = cv2.bitwise_not(imbin)
    imbin1 = imfill(imbin1)
    # Erode
    imbin1 = cv2.erode(imbin1, kernel=np.ones(
        (3, 3), dtype=np.uint8), anchor=(-1, -1), iterations=10)

    h, w = imbin1.shape
    # Add margins for erosion
    for col in range(w):
        imbin1[0, col] = 0
        imbin1[h - 1, col] = 0
    for row in range(h):
        imbin1[row, 0] = 0
        imbin1[row, w - 1] = 0
    imbin1 = cv2.erode(imbin1, kernel=np.ones(
        (3, 3), dtype=np.uint8), anchor=(-1, -1), iterations=4)
    imbin1 = imfill(imbin1)

    # Mask validation
    # The stats var is [num_of_connected_component, 5 (left, top, width, height, area)]
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(
        imbin1, connectivity=8, ltype=cv2.CV_16U)

    # No leaf pixels found
    if retval < 2:
        print('No leaf pixels found. No mask.')
        return None

    # Leaf area so small, skip
    big_l = 1
    for i in range(1, retval):
        if stats[i][4] > stats[big_l][4]:
            big_l = i
    if stats[big_l][4] < (h * w * 0.15):
        print('Leaf area so small! No mask.')
        return None

    coor_x = coor_y = 0
    # Delete other objects, keep the biggest
    for row in range(aux_height):
        for col in range(aux_width):
            imbin1[row, col] = 255 if labels[row, col] == big_l else 0

    imbin1 = cv2.erode(imbin1, kernel=np.ones(
        (3, 3), dtype=np.uint8), anchor=(-1, -1), iterations=10)
    imbin1 = cv2.resize(imbin1, dsize=(width, height), fx=0, fy=0,
                        interpolation=cv2.INTER_NEAREST)

    return imbin1


def imfill(inmask):
    """
        Do floodFill operation on the input mask
    """
    flood_mask = inmask.copy()
    h, w = flood_mask.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    for col in range(w):
        if flood_mask[0, col] == 0:
            cv2.floodFill(flood_mask, mask, (col, 0),
                          newVal=255, loDiff=10, upDiff=10)
        if flood_mask[h - 1, col] == 0:
            cv2.floodFill(flood_mask, mask, (col, h - 1),
                          newVal=255, loDiff=10, upDiff=10)

    for row in range(h):
        if flood_mask[row, 0] == 0:
            cv2.floodFill(flood_mask, mask, (0, row),
                          newVal=255, loDiff=10, upDiff=10)
        if flood_mask[row, w - 1] == 0:
            cv2.floodFill(flood_mask, mask, (w - 1, row),
                          newVal=255, loDiff=10, upDiff=10)

    out_mask = inmask.copy()

    for row in range(h):
        for col in range(w):
            if flood_mask[row, col] == 0:
                out_mask[row, col] = 255

    return out_mask


def on_focus(imask, th=0.7):
    """
        Check image patches focused or not
    Args:
        imask:  Image patches' mask
    """
    mask_ratio = np.mean(imask)
    if mask_ratio > th:
        focused = True
    else:
        focused = False

    return focused


if __name__ == '__main__':

    from PIL import Image
    from matplotlib import pyplot as plt

    plt.rc('font', size=12)
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('figure', titlesize=12)

    timestamp = '07-06-21'
    tray = 'tray1'
    leaf_disk_filename = '1'

    img = Image.open(
        f'/media/cornell/Data/tq42/Hyphal_2020/data/{timestamp}/{leaf_disk_filename}.tif')
    imask = leaf_mask(img, rel_th=0.15)

    img_arr = np.asarray(img)
    imask_arr = np.expand_dims(imask, axis=2)
    imask_arr = np.repeat(imask_arr, repeats=3, axis=2)

    masked_img_arr = img_arr * imask_arr
    img_arr_copy = img_arr.copy()
    img_arr_copy[imask == 0] = 0

    plt_fig, plt_axis = plt.subplots(figsize=(10, 6))

    plt_axis.imshow(img_arr)
    plt_axis.set_title('Raw leaf disk image')
    plt_axis.axis('off')

    # plt_axis[0].imshow(img_arr)
    # plt_axis[0].set_title('Raw leaf disk image')
    # plt_axis[0].get_xaxis().set_visible(False)
    # plt_axis[0].get_yaxis().set_visible(False)


    # plt_axis[1].imshow(img_arr_copy)
    # plt_axis[1].set_title('Masked leaf disk image')
    # plt_axis[1].get_xaxis().set_visible(False)
    # plt_axis[1].get_yaxis().set_visible(False)

    plt_fig.savefig(f'{timestamp}_{leaf_disk_filename}.png', dpi=300)


    plt_fig, plt_axis = plt.subplots(figsize=(10, 6))

    plt_axis.imshow(img_arr_copy)
    plt_axis.set_title('Masked leaf disk image')
    plt_axis.axis('off')

    # plt_axis[0].imshow(img_arr)
    # plt_axis[0].set_title('Raw leaf disk image')
    # plt_axis[0].get_xaxis().set_visible(False)
    # plt_axis[0].get_yaxis().set_visible(False)


    # plt_axis[1].imshow(img_arr_copy)
    # plt_axis[1].set_title('Masked leaf disk image')
    # plt_axis[1].get_xaxis().set_visible(False)
    # plt_axis[1].get_yaxis().set_visible(False)

    plt_fig.savefig('masked.png', dpi=300)
