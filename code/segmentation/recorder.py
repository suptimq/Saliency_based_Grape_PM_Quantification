import torch


class Recorder:
    def __init__(self, name):
        """
            The class to record loss history in the training process 
        """
        self._name = name

        self._loss = 0
        self.correct_counts = 0
        self.total_counts = 0
        self.total_images = 0

        self.pred_infected_ = 0
        self.gt_infected_ = 0
        self.correct_infected_ = 0

    def update(self, preds, labels, loss):
        self.correct_counts += self.count_correct(preds, labels)
        self.total_counts += labels.nelement()
        self.total_images += len(labels)
        self._loss += loss

        pred_mask = torch.argmax(preds, dim=1).detach()
        labels_copy = labels.detach().clone()
        pred_mask[pred_mask == 0] = -1
        labels_copy[labels_copy == 0] = -2
        self.pred_infected_ += len(pred_mask[pred_mask == 1])
        self.gt_infected_ += len(labels_copy[labels_copy == 1])
        self.correct_infected_ += (pred_mask==labels_copy).sum().item()

    @property
    def name(self):
        return self._name

    @property
    def loss(self):
        return self._loss

    @property
    def correct(self):
        return self.correct_counts

    @property
    def total(self):
        return self.total_counts

    @property
    def total_image(self):
        return self.total_images

    @property
    def pred_infected(self):
        return self.pred_infected_

    @property
    def gt_infected(self):
        return self.gt_infected_

    @property
    def correct_infected(self):
        return self.correct_infected_

    def reset(self):
        self._loss = 0
        self.correct_counts = 0
        self.total_counts = 0
        self.total_images = 0

        self.pred_infected_ = 0
        self.gt_infected_ = 0
        self.correct_infected_ = 0

    def count_correct(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).sum().item()
