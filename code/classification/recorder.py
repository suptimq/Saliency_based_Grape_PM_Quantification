import torch


class Recorder:
    def __init__(self, name):
        """
            The class to record loss history in the training process 
        """
        self._loss = 0
        self.correct_counts = 0
        self.total_counts = 0
        self._name = name

    def update(self, preds, labels, loss):
        self.correct_counts += self.count_correct(preds, labels)
        self.total_counts += len(labels)
        self._loss += loss

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
    
    def reset(self):
        self._loss = 0
        self.correct_counts = 0
        self.total_counts = 0

    def count_correct(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).sum().item()