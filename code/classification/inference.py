import os
import h5py
import glob
import argparse
import numpy as np
import pandas as pd

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
    label_ds = f['labels']
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


def pred_img(img, model):
    """
        Get predicted image class and prob using well-trained model
    Args:
        img: PIL image or np.ndarray
    """

    out = model(img)

    pred = torch.argmax(out, axis=1)
    prob = F.softmax(out, dim=1)

    return pred, prob


def categorize(pred, true, idx, t_p, t_n, f_p, f_n):
    """
        Categorize each predicted result into one of four categories in a confusion matrix
    """
    status = "Correct"
    if pred == 0:
        if pred == true:
            t_n.append(idx)
        else:
            f_n.append(idx)
            status = "Incorrect"
    else:
        if pred == true:
            t_p.append(idx)
        else:
            f_p.append(idx)
            status = "Incorrect"

    return status


if __name__ == "__main__":
    from utils import init_model, load_model, parse_model, plot_confusion_matrix

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
                        default='GoogleNet',
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
    parser.add_argument('--cv_dai', type=str,
                        help='date to be tested')
    parser.add_argument('--cv_qtl', type=str,
                        help='qtl partition to be tested')
    parser.add_argument('--cv_seg_dataset', type=str,
                        help='seg dataset to be tested')
    parser.add_argument('--set', default='val', choices=['train', 'val', 'test'],
                        help='use train/val/test set for inference')
    opt = parser.parse_args()

    model_para = parse_model(opt)
    dataset_root_path = Path(opt.dataset_path) / 'data'
    image_folder = dataset_root_path / opt.img_folder

    output_folder = Path(
        os.getcwd()).parent / 'results' / 'journal'

    means = [116./255., 156./255., 80./255.]
    stds = [38./255., 34./255., 48./255.]
        
    subfolder_name = 'inference_results'
    if opt.set == 'train':
        test_filepath = r'train_set.hdf5'
    elif opt.set == 'val':
        test_filepath = r'val_set.hdf5'
        if opt.cv_dai:
            dataset_root_path = dataset_root_path / 'cross_validation_ds' / opt.cv_dai
            subfolder_name = f'cross_validation/{opt.cv_dai}'
        if opt.cv_qtl:
            dataset_root_path = dataset_root_path / 'qtl_partition_test' / f'partition_{opt.cv_qtl}'
            subfolder_name = f'cross_validation/partition_{opt.cv_qtl}'
        if opt.cv_seg_dataset:
            dataset_root_path = dataset_root_path / 'segmentation' / 'cls_dataset' / f'cv{opt.cv_seg_dataset}'
            subfolder_name = f'cross_validation/seg_dataset_{opt.cv_seg_dataset}'
    elif opt.set == 'test':
        test_filepath = r'test_set.hdf5'
        means = [118./255., 165./255., 92./255.]
        stds = [40./255., 35./255., 51./255.]

    output_folder = output_folder / subfolder_name / opt.group / opt.model_type

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    dataset_para = {
        'dataset_folder': dataset_root_path,
        'test_filepath': test_filepath,
        'image_folder': image_folder
    }
    printArgs(None, vars(opt))

    if opt.HDF5:
        # Import data from HDF5
        images, labels = load_f5py(dataset_para)

    else:
        # Import data from image direcotry
        images, image_filenames, labels = load_dir(dataset_para)

    if opt.model_type == 'Inception3':
        test_transform = tvtrans.Compose([
            tvtrans.ToPILImage(),
            tvtrans.Resize(299),
            tvtrans.ToTensor(),
            tvtrans.Normalize(means, stds)
        ])
    else:
        test_transform = tvtrans.Compose([
            tvtrans.ToTensor(),
            tvtrans.Normalize(means, stds)
        ])

    # Load model
    cuda_id = model_para['cuda_id']
    model, device = load_model(model_para)
    model.eval()

    # Write CSV
    META_COL_NAMES = ['id', 'predicted class', 'true class', 'status']
    pred_df = pd.DataFrame(columns=META_COL_NAMES)

    print("INFERENCE START")

    correct_counts, total_counts = 0, 0
    infected_counts, clear_counts = 0, 0
    class_name_map = {0: 'Clear',  1: 'Infected'}
    # Confusion matrix
    y_true = []
    y_pred = []
    # Positive: infected, negative: clear
    f_n = []
    f_p = []
    t_n = []
    t_p = []

    for idx in range(images.shape[0]):
        cur_img = images[idx]
        preproc_img = test_transform(cur_img).unsqueeze(0).to(device)
        cur_label = labels[idx]
        y_true.append(cur_label[0])
        # cur_filename = image_filenames[idx]
        pred, prob = pred_img(preproc_img, model)
        pred = pred.cpu().detach().item()
        y_pred.append(pred)
        status = categorize(pred, cur_label[0], idx, t_p, t_n, f_p, f_n)
        record_df = pd.DataFrame(
            [[idx, class_name_map[pred], class_name_map[cur_label[0]], status]], columns=META_COL_NAMES)

        correct_counts += 1 if pred == cur_label[0] else 0
        total_counts += 1
        infected_counts += 1 if cur_label[0] == 1 else 0
        clear_counts += 1 if cur_label[0] == 0 else 0

        pred_df = pred_df.append(record_df, ignore_index=True)

        # print('Image idx: {0}\tCorrect label: {1}\tPredicted label: {2}'.format(
        #     idx, cur_label[0], pred))

    accuracy = 100.0 * correct_counts / total_counts
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    plot_confusion_matrix(
        output_folder, cm,
        list(class_name_map.values()),
        normalize=True,
        filename=f'confusion-matrix-{opt.model_type}-{opt.set}-{opt.loading_epoch}.png',
        title=f'Confusion Matrix\nOverall Accuracy: {accuracy:.2f}%\nF1 Score: {f1:.2f}')
    print('Accuracy of the network on the {0} val images: {1:.3f}%'.format(
        total_counts, accuracy))
    print('Clear {} infected: {}'.format(clear_counts, infected_counts))

    # print('INFERENCE FINISHED')

    # np.random.shuffle(t_p)
    # np.random.shuffle(t_n)
    # np.random.shuffle(f_p)
    # np.random.shuffle(f_n)
    # print('True infected patches\' idx : {}'.format(t_p[:10]))
    # print('False infected patches\' idx : {}'.format(f_p[:10]))
    # print('True clear patches\' idx : {}'.format(t_n[:10]))
    # print('False clear patches\' idx : {}'.format(f_n[:10]))
