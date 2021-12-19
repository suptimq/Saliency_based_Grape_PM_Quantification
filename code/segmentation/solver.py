### Solver (train and test)
import os
import time
import copy
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from recorder import Recorder
from utils import makeSubdir, logInfoWithDot, timeSince, init_model, init_optimizer


class Solver():
    def __init__(self, model, dataloader, optimizer, scheduler, logger, writer):
        """
        Args:
            model:         Model params
            dataloader:    Dataloader params
            optimizer:     Optimizer params
            scheduler:     Scheduler params
            logger:        Logger params
            writer:        Tensorboard writer params
        """
        self.model = init_model(model)
        self.pretrained = model['pretrained']
        self.loading_epoch = model['loading_epoch']
        self.total_epochs = model['total_epochs']
        self.model_path = model['model_path']
        self.model_filename = model['model_filename']
        self.save = model['save']
        self.model_name = model['model_type']
        self.model_fullpath = str(self.model_path / self.model_filename)

        self.init_random_seed = model['manual_seed']

        # Best model
        self.is_best = False
        self.best_model = None
        self.best_acc = 0
        self.best_iou = 0
        self.best_dice = 0
        self.best_optim = None
        self.best_epoch = 0
        self.best_model_filepath = str(
            self.model_path / 'best_model_checkpoint.pth.tar')

        # Logger
        self.logger = logger
        # Writer: Default path is runs/CURRENT_DATETIME_HOSTNAME
        writer_fullname = writer['writer_path'] / writer['writer_filename']
        self.writer = SummaryWriter(log_dir=writer_fullname)

        if model['gpu'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
            logInfoWithDot(self.logger, "USING GPU")
        else:
            self.device = torch.device('cpu')
            logInfoWithDot(self.logger, "USING CPU")

        if model['resume']:
            load_model_path = self.model_fullpath.format(
                self.model_name, self.loading_epoch)
            checkpoint = torch.load(load_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Log information
            logInfoWithDot(self.logger, "LOADING MODEL FINISHED")

        if optimizer['weighted_loss']:
            weights = torch.tensor([1., 10.]).to(self.device)
            self.logger.info(f'weighted {weights}')
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = init_optimizer(optimizer, self.model)
        if optimizer['resume']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Log information
            logInfoWithDot(self.logger, "LOADING OPTIMIZER FINISHED")

        # Scheduler
        self.scheduler = None
        if scheduler['use']:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler['step_size'],
                gamma=scheduler['gamma'])

        # Dataloader
        self.trainloader = dataloader['train']
        self.validloader = dataloader['valid']

        # Evaluation
        self.train_recorder = Recorder('train')
        self.test_recorder = Recorder('test')

        # Timer
        self.start_time = time.time()

    def train_one_epoch(self, ep, log_interval):
        pass

    def test_one_epoch(self, ep):
        pass

    def forward(self, log_interval):
        start_epoch = self.loading_epoch + 1
        total_epochs = start_epoch + self.total_epochs
        for epoch in range(start_epoch, total_epochs):

            np.random.seed(self.init_random_seed+epoch)

            self.train_one_epoch(epoch, log_interval=log_interval)
            self.test_one_epoch(epoch)

            # Update learning rate after every specified times iteration
            if self.scheduler:
                self.scheduler.step()

            if self.save and self.is_best:
                makeSubdir(self.model_path)
                torch.save({
                    'epoch': self.best_epoch,
                    'model_state_dict': self.best_model.state_dict(),
                    'optimizer_state_dict': self.best_optim.state_dict(),
                    'best_acc': self.best_acc,
                }, self.best_model_filepath)

                # Log information
                logInfoWithDot(
                    self.logger, "SAVED MODEL: {}".format(
                        self.model_fullpath.format(self.model_name, self.best_epoch)))

        # Rename
        shutil.move(self.best_model_filepath, self.model_fullpath.format(
            self.model_name, self.best_epoch))

        # self.writer.add_graph(self.model)
        self.writer.close()
        logInfoWithDot(
            self.logger, "TRAINING FINISHED, TIME USAGE: {} secs".format(
                timeSince(self.start_time)))


class HyphalSolver(Solver):
    def train_one_epoch(self, ep, log_interval=10):
        self.model.train()

        self.train_recorder.reset()

        lr = self.optimizer.param_groups[0]['lr']

        for i, (images, labels) in enumerate(self.trainloader, 0):
            images = images.to(self.device)
            labels = labels.squeeze(1).to(self.device).long()

            preds = self.model(images)

            loss = self.criterion(preds['out'], labels)

            # Update
            self.train_recorder.update(preds['out'], labels, loss.item())
            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % log_interval == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLearning Rate: {}\tLoss: {:.6f}\tTime Usage:{:.8}'
                    .format(ep, i * len(images), len(self.trainloader.dataset),
                            100. * i / len(self.trainloader), lr, loss.data,
                            timeSince(self.start_time)))

            if i == len(self.trainloader) - 1:
                pixel_accuracy = 100.0 * self.train_recorder.correct / self.train_recorder.total
                iou_accuracy = 100.0 * self.train_recorder.correct_infected / \
                    (self.train_recorder.pred_infected +
                     self.train_recorder.gt_infected - self.train_recorder.correct_infected)
                dice_accuracy = 100.0 * 2 * self.train_recorder.correct_infected / \
                    (self.train_recorder.pred_infected +
                     self.train_recorder.gt_infected)
                self.logger.info(
                    'Loss of the network on the {0} train images: {1:.6f}'.
                    format(self.train_recorder.total_image, self.train_recorder.loss))
                self.logger.info(
                    'Pixel accuracy of the network on the {0} train images: {1:.3f}%'
                    .format(self.train_recorder.total_image, pixel_accuracy))
                self.logger.info(
                    'IOU accuracy of the network on the {0} train images: {1:.3f}%'
                    .format(self.train_recorder.total_image, iou_accuracy))
                self.logger.info(
                    'Dice accuracy of the network on the {0} train images: {1:.3f}%'
                    .format(self.train_recorder.total_image, dice_accuracy))

                # Write to Tensorboard file
                self.writer.add_scalar(
                    'Loss/train', self.train_recorder.loss, ep)
                self.writer.add_scalar(
                    'Pixel_Accuracy/train', pixel_accuracy, ep)
                self.writer.add_scalar('IOU_Accuracy/train', iou_accuracy, ep)
                self.writer.add_scalar(
                    'Dice_Accuracy/train', dice_accuracy, ep)

    def test_one_epoch(self, ep):
        self.model.eval()

        self.test_recorder.reset()

        for _, (images, labels) in enumerate(self.validloader, 0):
            images = images.to(self.device)
            labels = labels.squeeze(1).to(self.device).long()

            preds = self.model(images)
            loss = self.criterion(preds['out'], labels)

            self.test_recorder.update(preds['out'], labels, loss.item())

        pixel_accuracy = 100.0 * self.test_recorder.correct / self.test_recorder.total
        iou_accuracy = 100.0 * self.test_recorder.correct_infected / \
            (self.test_recorder.pred_infected +
             self.test_recorder.gt_infected - self.test_recorder.correct_infected)
        dice_accuracy = 100.0 * 2 * self.test_recorder.correct_infected / \
            (self.test_recorder.pred_infected + self.test_recorder.gt_infected)
        self.logger.info(
            'Loss of the network on the {0} val images: {1:.6f}'.format(
                self.test_recorder.total_image, self.test_recorder.loss))
        self.logger.info(
            'Pixel accuracy of the network on the {0} val images: {1:.3f}%'
            .format(self.test_recorder.total_image, pixel_accuracy))
        self.logger.info(
            'IOU accuracy of the network on the {0} val images: {1:.3f}%'
            .format(self.test_recorder.total_image, iou_accuracy))
        self.logger.info(
            'Dice accuracy of the network on the {0} val images: {1:.3f}%'
            .format(self.test_recorder.total_image, dice_accuracy))

        # Compare accuracy
        self.is_best = iou_accuracy > self.best_iou
        if self.is_best:
            self.best_acc = pixel_accuracy
            self.best_iou = iou_accuracy
            self.best_dice = dice_accuracy
            self.best_model = copy.deepcopy(self.model)
            self.best_optim = copy.deepcopy(self.optimizer)
            self.best_epoch = ep

        # Write to Tensorboard file
        self.writer.add_scalar('Loss/val', self.test_recorder.loss, ep)
        self.writer.add_scalar('Pixel_Accuracy/val', pixel_accuracy, ep)
        self.writer.add_scalar('IOU_Accuracy/val', iou_accuracy, ep)
        self.writer.add_scalar('Dice_Accuracy/val', dice_accuracy, ep)
