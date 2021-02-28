"""
Includes all utils related to training
"""

import torch

from typing import Dict
from torch import Tensor
from omegaconf import DictConfig


def compute_soft_accuracy_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Calculate multiclass accuracy with logits (one class also works)- soft accuracy
    :param logits: tensor with logits from the model
    :param labels: tensor holds all the labels with their scores: (0, 1/3, 2/3, 1)
    :return: score for each sample
    """
    logits = torch.max(logits, 1)[1].data  # argmax- matrix with the label with max prob. [batch_size, 1]

    logits_one_hots = torch.zeros(*labels.size()) # zeros matrix- [batch_size, num_ans]
    if torch.cuda.is_available():
        logits_one_hots = logits_one_hots.cuda()
    logits_one_hots.scatter_(1, logits.view(-1, 1), 1)# one hot matrix- [batch_size, num_ans]

    scores = (logits_one_hots * labels) # soft accuracy

    return scores


def compute_accuracy_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Calculate multiclass accuracy with logits (one class also works)- accuracy
    :param logits: tensor with logits from the model
    :param labels: tensor holds all the labels with their scores: (0, 1/3, 2/3, 1)
    :return: score for each sample
    """

    num_ans = labels.size()
    logits = torch.max(logits, 1)[1].data  # argmax- matrix with the label with max prob. [batch_size, 1]
    logits_one_hots = torch.zeros(*num_ans) # zeros matrix- [batch_size, num_ans]
    if torch.cuda.is_available():
        logits_one_hots = logits_one_hots.cuda()
    logits_one_hots.scatter_(1, logits.view(-1, 1), 1)# one hot matrix- [batch_size, num_ans]

    argmax_labels = torch.max(labels, 1)[1].data  # argmax- matrix with the label with max prob. [batch_size, 1]
    labels_one_hots = torch.zeros(*num_ans)  # zeros matrix- [batch_size, num_ans]
    if torch.cuda.is_available():
        labels_one_hots = labels_one_hots.cuda()
    labels_one_hots.scatter_(1, argmax_labels.view(-1, 1), 1)  # one hot matrix- [batch_size, num_ans]
    for i, (val, idx) in enumerate(zip(torch.max(labels, 1)[0].data, torch.max(labels, 1)[1].data)):
        if val == 0:
            labels_one_hots[i][idx] = 0.0

    scores = (logits_one_hots * labels_one_hots) # accuracy

    return scores



def get_zeroed_metrics_dict() -> Dict:
    """
    :return: dictionary to store all relevant metrics for training
    """
    return {'train_loss': 0, 'train_soft_acc': 0, 'train_acc': 0, 'total_norm': 0, 'count_norm': 0}


class TrainParams:
    """
    This class holds all train parameters.
    Add here variable in case configuration file is modified.
    """
    num_epochs: int
    lr: float
    lr_decay: float
    lr_gamma: float
    lr_step_size: int
    grad_clip: float
    save_model: bool

    def __init__(self, **kwargs):
        """
        :param kwargs: configuration file
        """
        self.num_epochs = kwargs['num_epochs']

        self.lr = kwargs['lr']['lr_value']
        self.lr_decay = kwargs['lr']['lr_decay']
        self.lr_gamma = kwargs['lr']['lr_gamma']
        self.lr_step_size = kwargs['lr']['lr_step_size']

        self.grad_clip = kwargs['grad_clip']
        self.save_model = kwargs['save_model']


def get_train_params(cfg: DictConfig) -> TrainParams:
    """
    Return a TrainParams instance for a given configuration file
    :param cfg: configuration file
    :return:
    """
    return TrainParams(**cfg['train'])