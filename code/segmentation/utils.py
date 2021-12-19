import os
import sys
import time
import math
import socket
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('figure', titlesize=12)


def plot_confusion_matrix(output_folder, cm, classes, normalize=False, filename='confusion-matrix.png', title='Confusion Matrix', cmap=plt.cm.Blues):
    """
        Plot confusion matrix
    Args:
        output_folder:
        cm:               Confusion matrix ndarray
        classes:          A list of class
    """
    cm_copy = cm.copy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.text(j, i+0.1, format(cm_copy[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    output_filepath = output_folder / filename
    plt.savefig(output_filepath)


def set_logging(log_file, log_level=logging.DEBUG):
    """
    Logging to console and log file simultaneously.
    """
    log_format = '[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s'
    formatter = logging.Formatter(log_format)
    logging.basicConfig(level=log_level, format=log_format, filename=log_file)
    # Console Log Handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(log_level)
    logging.getLogger().addHandler(console)
    return logging.getLogger()


def getTimestamp(f='%b%d_%H-%M-%S'):
    return datetime.now().strftime(f)


def makeSubdir(dirname):
    # Recursively make directories
    os.makedirs(dirname, exist_ok=True)


def getHostName():
    return socket.gethostname()


def printArgs(logger, args):
    for k, v in args.items():
        if logger:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))


def logInfoWithDot(logger, text):
    logger.info(text)
    logger.info('--------------------------------------------')


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))


"""
------------PyTorch------------
"""


def parse_model(opt):
    result_root_path = Path(opt.model_path) / 'results'

    current_time = opt.timestamp
    model_type_time = opt.model_type + '_{}'.format(current_time)

    model_path = result_root_path / 'models' / model_type_time
    model_filename = '{0}_model_ep{1:03}'

    pretrained = vars(opt).get('pretrained', False)
    cuda_id = opt.cuda_id if opt.cuda and torch.cuda.is_available() else None
    model_para = {
        'model_type': opt.model_type,
        'pretrained': pretrained,
        'outdim': opt.outdim,
        'model_path': model_path,
        'model_filename': model_filename,
        'loading_epoch': opt.loading_epoch,
        'cuda': opt.cuda,
        'cuda_id': cuda_id
    }

    return model_para


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_model(model_para):
    """
        Load well-trained model
    """
    model = init_model(model_para)
    model_fullpath = str(
        model_para['model_path'] / model_para['model_filename'])

    cuda_id = model_para['cuda_id']
    device = torch.device(
        f'cuda:{cuda_id}' if cuda_id else 'cpu')

    checkpoint = torch.load(model_fullpath.format(
        model_para['model_type'], model_para['loading_epoch']), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.to(device), device


def init_model(model):
    m = None
    outdim = model['outdim']
    pretrained = model.get('pretrained', False)
    feature_extract = model.get('feature_extract', False)

    if model['model_type'] == 'FCN':
        m = models.segmentation.fcn_resnet50(pretrained=pretrained)
        set_parameter_requires_grad(m, feature_extract)
        m.classifier = models.segmentation.fcn.FCNHead(2048, outdim)
    elif model['model_type'] == 'DeepLab':
        m = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        set_parameter_requires_grad(m, feature_extract)
        m.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, outdim)

    assert m != None, 'Model Not Initialized'
    return m


def init_optimizer(optimizer, model):
    opt = None
    lr = optimizer['lr']
    weight_decay = optimizer['weight_decay']
    parameters = model.parameters()
    if optimizer['optim_type'] == 'Adam':
        opt = optim.Adam(parameters,
                         lr=lr,
                         weight_decay=weight_decay)
    elif optimizer['optim_type'] == 'Adadelta':
        opt = optim.Adadelta(parameters,
                             lr=lr,
                             weight_decay=weight_decay)
    elif optimizer['optim_type'] == 'SGD':
        opt = optim.SGD(parameters,
                        lr=lr,
                        weight_decay=weight_decay)
    else:
        opt = optim.RMSprop(parameters,
                            lr=lr,
                            weight_decay=weight_decay)

    assert opt != None, 'Optimizer Not Initialized'
    return opt
