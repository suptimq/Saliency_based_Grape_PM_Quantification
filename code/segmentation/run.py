import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torchvision.transforms as tvtrans
import tensorboard

from utils import (getTimestamp, getHostName, makeSubdir,
                   logInfoWithDot, printArgs, set_logging)
from dataloader import HyphalDataset
from solver import HyphalSolver

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--model_type',
                    default='DeepLab',
                    help='model used for training',
                    choices=['DeepLab', 'FCN'])
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model parameters')
parser.add_argument('--feature_extract', action='store_true',
                    help='fine-tune the last layer only')
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--resume_timestamp', help='timestamp to resume')
parser.add_argument('--loading_epoch',
                    type=int,
                    default=0,
                    help='xth model loaded to resume')
parser.add_argument('--total_epochs',
                    type=int,
                    default=200,
                    help='number of epochs to train for')
parser.add_argument('--outdim', type=int, default=2, help='number of classes')
parser.add_argument('--save_model', action='store_true', help='save model')
parser.add_argument('--cuda', action='store_true', help='enable cuda')

# Optimizer
parser.add_argument('--optimType',
                    default='Adam',
                    help='optimizer used for training',
                    choices=['Adam', 'Adadelta', 'RMSprop', 'SGD'])
parser.add_argument('--lr',
                    type=float,
                    default=1e-4,
                    help='learning rate for optimzer')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.01,
                    help='weight decay for optimzer')
parser.add_argument('--weighted_loss',
                    action='store_true',
                    help='weighted loss')

# Scheduler
parser.add_argument('--scheduler', action='store_true', help='use scheduler')
parser.add_argument('--step_size',
                    type=int,
                    default=60,
                    help='period of learning rate decay')
parser.add_argument('--gamma',
                    type=float,
                    default=0.5,
                    help='multiplicative factor of learning rate decay')

# Dataloader
parser.add_argument('--bsize', type=int, default=1, help='input batch size')
parser.add_argument('--nworker',
                    type=int,
                    default=4,
                    help='number of dataloader workers')

parser.add_argument('--manual_seed',
                    type=int,
                    default=1701,
                    help='reproduce experiemnt')
parser.add_argument('--cuda_device',
                    default="0",
                    help='ith cuda used for training')
parser.add_argument('--root_path', type=str,
                    required=True, help='path to data')
parser.add_argument('--cls_seg_dataset', type=str,
                    help='use masks generated from classification model')
parser.add_argument('--cv', action='store_true',
                    help='cross validation')
parser.add_argument('--seg_idx', type=str,
                    help='num. cv')
parser.add_argument('--randomized_dataset', action='store_true',
                    help='use randomized dataset')
parser.add_argument('--aug_dataset', action='store_true',
                    help='use augmented dataset')
opt = parser.parse_args()

np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
torch.cuda.manual_seed(opt.manual_seed)

gpu = opt.cuda

if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device

# Basic configuration
root_path = Path(opt.root_path)
result_root_path = root_path / 'results'
dataset_root_path = root_path / 'data' / 'segmentation'

if opt.cls_seg_dataset:
    meta_filepath = 'metadata.csv'
    train_filepath = f'cls_seg_mask/train_set_{opt.cls_seg_dataset}.hdf5'
    val_filepath = 'val_set_all.hdf5'
elif opt.cv:
    meta_filepath = 'metadata.csv'
    dataset_root_path = dataset_root_path / f'cls_dataset/cv{opt.seg_idx}'
    train_filepath = f'train_set.hdf5'
    val_filepath = 'val_set.hdf5'
elif opt.randomized_dataset:
    meta_filepath = 'metadata.csv'
    train_filepath = 'train_set_randomization.hdf5'
    val_filepath = 'val_set.hdf5'
elif opt.aug_dataset:
    meta_filepath = 'metadata.csv'
    train_filepath = 'train_set_aug.hdf5'
    val_filepath = 'val_set_aug.hdf5'
else:
    meta_filepath = 'metadata.csv'
    train_filepath = 'train_set.hdf5'
    val_filepath = 'val_set.hdf5'


# Dataloader
bsize = opt.bsize
nworker = opt.nworker

# Model
year = '2021'
current_time = getTimestamp() if not opt.resume else opt.resume_timestamp
model_type_time = opt.model_type + '_{0}_{1}'.format(current_time, year)

model_path = result_root_path / 'models' / model_type_time
model_filename = '{0}_model_ep{1:03}'

# Logger and Writer
log_path = result_root_path / 'logs' / 'seg' / model_type_time
log_filename = 'log_{0}_{1}_{2}.txt'.format(
    'train', current_time, year)  # log_train/test_currentTime
writer_path = result_root_path / 'runs' / 'seg'
writer_filename = '{0}_{1}_{2}_{3}'.format(
    opt.model_type, current_time, year, getHostName())  # modelName_currentTime_hostName

# Parameters for dataset
dataset_path = {
    'root_path': dataset_root_path,
    'meta_filepath': meta_filepath,
    'train_filepath': train_filepath,
    'val_filepath': val_filepath
}

# Parameters for solver
model = {
    'model_type': opt.model_type,
    'pretrained': opt.pretrained,
    'outdim': opt.outdim,
    'resume': opt.resume,
    'loading_epoch': opt.loading_epoch,
    'total_epochs': opt.total_epochs,
    'model_path': model_path,
    'model_filename': model_filename,
    'save': opt.save_model,
    'gpu': gpu,
    'feature_extract': opt.feature_extract,
    'manual_seed': opt.manual_seed
}

optimizer = {
    'optim_type': opt.optimType,
    'resume': opt.resume,
    'lr': opt.lr,
    'weight_decay': opt.weight_decay,
    'weighted_loss': opt.weighted_loss
}

scheduler = {
    'use': opt.scheduler,
    'step_size': opt.step_size,
    'gamma': opt.gamma
}

logging = {
    'log_path': log_path,
    'log_filename': log_filename,
    'log_level': 20,  # 20 == level (logging.INFO)
}

writer = {'writer_path': writer_path, 'writer_filename': writer_filename}

# Log config
makeSubdir(logging['log_path'])
logger = set_logging(logging['log_path'] / logging['log_filename'],
                     logging['log_level'])


# Log model, optim information
printArgs(logger, vars(opt))

## Preprocessing transforms: data augmentation
# 18000 samples
#  train mean (116.83, 156.61, 79.98) std (38.75, 34.61, 48.77)
#  val mean (116.05, 155.6, 78.77) std (38.31, 34.38, 48.1)

# train_augmentation = tvtrans.Compose([
#     tvtrans.ToPILImage(),
#     # tvtrans.RandomHorizontalFlip(p=0.5),
#     # tvtrans.RandomVerticalFlip(p=0.5),
#     # tvtrans.RandomAffine(degrees=(0, 180), translate=(0.05, 0.05), scale=(0.9, 1.1)),
#     tvtrans.ToTensor(),
#     tvtrans.Normalize((0.5, ), (0.5, ))
# ])
# test_transform = tvtrans.Compose([
#     # tvtrans.ToPILImage(),
#     tvtrans.ToTensor(),
#     tvtrans.Normalize((0.5, ), (0.5, ))
# ])
    
# logger.info(train_augmentation)

worker_init_fn = lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)

# Get Hyphal dataset
kernel = (5, 5)
hyphal_train_ds = HyphalDataset(dataset_path,
                                train=True,
                                normalize=True,
                                thicken=True,
                                kernel=kernel)
# In case batch norm layer won't work on the single sample
drop_last = True if len(hyphal_train_ds) % bsize == 1 else False
hyphal_train_dl = torch.utils.data.DataLoader(hyphal_train_ds,
                                              batch_size=bsize,
                                              drop_last=drop_last,
                                              num_workers=nworker,
                                              worker_init_fn=worker_init_fn,
                                              shuffle=True)

hyphal_test_ds = HyphalDataset(dataset_path,
                               train=False,
                               normalize=True,
                               thicken=True,
                               kernel=kernel)
drop_last = True if len(hyphal_test_ds) % bsize == 1 else False
hyphal_test_dl = torch.utils.data.DataLoader(hyphal_test_ds,
                                             batch_size=bsize,
                                             drop_last=drop_last,
                                             num_workers=nworker,
                                             shuffle=False)
printArgs(logger, {'thicken kernel': kernel})
dataloader = {'train': hyphal_train_dl, 'valid': hyphal_test_dl}

solver = HyphalSolver(model, dataloader, optimizer, scheduler, logger, writer)
solver.forward(log_interval=50)
