import os
import cv2
import h5py
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score

import torch
import torch.nn.functional as F
import torchvision.transforms as tvtrans


np.random.seed(2020)


def printArgs(logger, args):
    for k, v in args.items():
        if logger:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))


def load_f5py(dataset_para):
    """
        Load data from HDF5 files or image directory
    """
    f = h5py.File(dataset_para['dataset_folder'] /
                  dataset_para['test_filepath'], 'r')

    image_ds = f['images']
    images = image_ds[:, ]
    label_ds = f['masks']
    labels = label_ds[:]
    return images, labels


def load_dir(dataset_para):
    label_class_map = {'Clear': 0, 'Infected': 1}

    image_folder = dataset_para['image_folder']
    image_filenames = glob.glob(str(image_folder / '*.jpg'))
    num = len(image_filenames)
    images = np.ndarray(shape=(num, 224, 224, 3), dtype=np.uint8)
    labels = np.zeros(shape=(num, 1), dtype=np.uint8)
    image_filename_list = []
    # Example filename: 57-Horizon_69_Clear/Infected.jpg, '/' means or
    #                   1-B9_cinerea_127.jpg, not manually classified
    for i, image_filename in enumerate(image_filenames):
        # Get labels if possible
        image_filename = os.path.basename(image_filename)
        image_filename_text = os.path.splitext(image_filename)[0]
        filename_strs = image_filename_text.split('_')
        # Determine classified or not
        if filename_strs[-1] in list(label_class_map.keys()):
            labels[i] = label_class_map[filename_strs[-1]]

        image_filepath = image_folder / image_filename
        img = Image.open(image_filepath)
        images[i] = np.asarray(img)
        image_filename_list.append(image_filename_text)

    return images, image_filename_list, labels


if __name__ == "__main__":
    from utils import load_model, parse_model, plot_confusion_matrix

    """ Usge
    python inference.py 
            --model_type SqueezeNet 
            --loading_epoch 190 
            --timestamp Jun04_18-18-46 
            --dataset_path /Users/tim/Documents/Cornell/CAIR/BlackBird/Data/Hyphal_2019
            --model_path /Users/tim/Documents/Cornell/CAIR/BlackBird/Data/Hyphal_2019
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        default='DeepLab',
                        help='model used for training')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model parameters')
    parser.add_argument('--loading_epoch',
                        type=int,
                        required=True,
                        help='xth model loaded for inference')
    parser.add_argument('--timestamp', required=True, help='model timestamp')
    parser.add_argument('--outdim', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--cuda', action='store_true', help='enable cuda')
    parser.add_argument('--cuda_id',
                        default="1",
                        help='specify cuda')
    parser.add_argument('--img_folder', type=str,
                        default='images', help='image folder')
    parser.add_argument('--dataset_path', type=str,
                        required=True, help='path to data')
    parser.add_argument('--model_path', type=str,
                        required=True, help='path to model')
    parser.add_argument('--HDF5', type=bool, default=True,
                        help='path to model')
    parser.add_argument('--group', type=str, default='baseline',
                        help='exp group')
    parser.add_argument('--set', default='val', choices=['train', 'val', 'test'],
                        help='use train/val/test set for inference')
    parser.add_argument('--thicken', action='store_true',
                        help='thicken hyphal lines')
    parser.add_argument('--seg_idx', type=str,
                        help='num. cv')
    parser.add_argument('--erosion', action='store_true',
                        help='erose hyphal lines')
    opt = parser.parse_args()

    model_para = parse_model(opt)
    result_root_path = Path(opt.model_path) / 'results'
    dataset_root_path = Path(opt.dataset_path) / 'data' / 'segmentation' / 'cls_dataset' / f'cv{opt.seg_idx}'
    image_folder = dataset_root_path / opt.img_folder
    output_folder = Path(
        os.getcwd()).parent / 'results' / 'journal' / 'segmentation' / 'segmentation' / 'cross_validation' / f'cv{opt.seg_idx}' / 'figures'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if opt.set == 'train':
        test_filepath = r'train_set.hdf5'
    elif opt.set == 'val':
        test_filepath = r'val_set.hdf5'

    dataset_para = {
        'dataset_folder': dataset_root_path,
        'test_filepath': test_filepath,
        'image_folder': image_folder
    }
    printArgs(None, vars(opt))

    means = [118./255., 165./255., 92./255.]
    stds = [40./255., 35./255., 51./255.]

    if opt.HDF5:
        # Import data from HDF5
        images, labels = load_f5py(dataset_para)
        test_transform = tvtrans.Compose([
            tvtrans.ToTensor(),
            tvtrans.Normalize(means, stds)
        ])
    else:
        # Import data from image direcotry
        images, image_filenames, labels = load_dir(dataset_para)
        test_transform = tvtrans.Compose([
            tvtrans.ToTensor(),
            tvtrans.Normalize(means, stds)
        ])

    # Load model
    cuda_id = model_para['cuda_id']
    model, device = load_model(model_para)
    model.eval()

    # Write CSV
    META_COL_NAMES = ['id', 'classification', 'label',
                      'status', 'infected (pred)', 'infected (mask)', 'IOU', 'Dice']
    pred_df = pd.DataFrame(columns=META_COL_NAMES)

    default_cmap = LinearSegmentedColormap.from_list(
        'MyColor', ['green', 'white', 'red']
    )

    print("INFERENCE START")

    clear_counts = 0
    infected_counts = 0
    correct_counts = 0
    total_counts = 0
    correct_infected_counts = 0
    pred_infected = 0
    classification_correct_counts = 0
    class_name_map = {0: 'Clear',  1: 'Infected'}
    # Confusion matrix
    y_true = []
    y_pred = []
    y_true_patch = []
    y_pred_patch = []

    format_ = 'pdf'

    for idx in range(images.shape[0]):
        cur_img = images[idx]
        preproc_img = test_transform(cur_img).unsqueeze(0).to(device)
        cur_mask = labels[idx]

        if opt.thicken:
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(cur_mask, kernel, iterations=1)
            cur_mask = dilated_mask

        preds = model(preproc_img)
        pred_mask = torch.argmax(preds['out'], dim=1).cpu().numpy()[0]

        if opt.erosion:
            kernel = np.ones((5, 5), np.uint8)
            erosion_mask = cv2.erode(pred_mask.astype(np.uint8), kernel, iterations=1)
            pred_mask = erosion_mask

        # Visualization of infected samples
        # if len(cur_mask[cur_mask != 0]):
        #     f, axs = plt.subplots(1, 1)

        #     axs.imshow(pred_mask, cmap=default_cmap)

        #     axs.get_xaxis().set_visible(False)
        #     axs.get_yaxis().set_visible(False)
        #     axs.set_xticklabels([])
        #     axs.set_yticklabels([])

        #     # axs[0].imshow(cur_img)
        #     # axs[1].imshow(cur_mask)
        #     # axs[2].imshow(pred_mask)

        #     # axs[1].get_xaxis().set_visible(False)
        #     # axs[1].get_yaxis().set_visible(False)
        #     # axs[1].set_xticklabels([])
        #     # axs[1].set_yticklabels([])

        #     # axs[2].get_xaxis().set_visible(False)
        #     # axs[2].get_yaxis().set_visible(False)
        #     # axs[2].set_xticklabels([])
        #     # axs[2].set_yticklabels([])

        #     print(
        #         f'number of infected pixels in mask: {len(pred_mask[pred_mask!=0])}')

        #     # plt.tight_layout()
        #     output_filepath = output_folder / f'{idx}.{format_}'
        #     plt.savefig(output_filepath, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
        #     plt.close()

        # Patch
        # Mask
        mask_clear_pixels = len(cur_mask[cur_mask == 0])
        mask_infected_pixels = len(cur_mask[cur_mask != 0])
        # Pred
        pred_infected_pixels = len(pred_mask[pred_mask != 0])
        pred_correct_counts = (cur_mask == pred_mask).sum()

        dilated_mask_copy = cur_mask.copy()
        pred_mask_copy = pred_mask.copy()
        dilated_mask_copy[dilated_mask_copy == 0] = -1
        pred_mask_copy[pred_mask_copy == 0] = -2
        pred_correct_infected_counts = (
            dilated_mask_copy == pred_mask_copy).sum()

        clear_counts += mask_clear_pixels
        infected_counts += mask_infected_pixels
        correct_counts += pred_correct_counts
        total_counts += pred_mask.size
        correct_infected_counts += pred_correct_infected_counts
        pred_infected += pred_infected_pixels

        y_true += cur_mask.ravel().tolist()
        y_pred += pred_mask.ravel().tolist()

        classificatin = 1 if pred_infected_pixels != 0 else 0
        classificatin_label = 1 if mask_infected_pixels != 0 else 0
        classification_correct_counts += 1 if classificatin == classificatin_label else 0
        y_true_patch.append(classificatin_label)
        y_pred_patch.append(classificatin)

        if pred_infected_pixels == 0 and mask_infected_pixels == 0:
            iou = dice = 0
        else:
            iou = 100 * pred_correct_infected_counts / \
                (pred_infected_pixels + mask_infected_pixels -
                 pred_correct_infected_counts)
            dice = 100 * 2 * pred_correct_infected_counts / \
                (pred_infected_pixels + mask_infected_pixels)

        tmp_df = pd.DataFrame([[idx, classificatin, classificatin_label, classificatin ==
                                classificatin_label, pred_infected_pixels, mask_infected_pixels, iou, dice]], columns=META_COL_NAMES)
        pred_df = pred_df.append(tmp_df, ignore_index=True)

    # Segementation
    pixel_accuracy = 100.0 * correct_counts / total_counts
    iou_accuracy = 100 * correct_infected_counts / \
        (correct_infected_counts+infected_counts-correct_infected_counts)
    dice_accuracy = 100 * 2 * correct_infected_counts / \
        (correct_infected_counts + infected_counts)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    plot_confusion_matrix(
        output_folder, cm,
        list(class_name_map.values()),
        normalize=True,
        filename=f'confusion_matrix_{opt.model_type}_{opt.set}_{opt.loading_epoch}.png',
        title=f'Confusion Matrix\nOverall Pixel Accuracy: {pixel_accuracy:.2f}% IOU: {iou_accuracy:.2f}% Dice: {dice_accuracy:.2f}%\nF1 Score: {f1:.2f}')
    print('Pixel accuracy of the network on the {0} val images: {1:.3f}%'.format(
        total_counts, pixel_accuracy))
    print('IoU accuracy of the network on the {0} val images: {1:.3f}%'.format(
        total_counts, iou_accuracy))
    print('Dice accuracy of the network on the {0} val images: {1:.3f}%'.format(
        total_counts, dice_accuracy))
    print('Clear {} infected: {}'.format(clear_counts, infected_counts))

    # Classification
    accuracy = 100 * classification_correct_counts / (idx + 1)
    cm = confusion_matrix(y_true_patch, y_pred_patch)
    f1 = f1_score(y_true_patch, y_pred_patch, average='macro') * 100
    plot_confusion_matrix(
        output_folder, cm,
        list(class_name_map.values()),
        normalize=True,
        filename=f'confusion-matrix-{opt.model_type}-{opt.set}-{opt.loading_epoch}-cls.png',
        title=f'Confusion Matrix\nAccuracy:{accuracy:.2f} F1 Score: {f1:.2f}')

    print('INFERENCE FINISHED')

    pred_df.to_csv(output_folder/'inference.csv', index=False)
    print('Saved {}'.format(output_folder/'inference.csv'))
