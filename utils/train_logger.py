"""
Train logger class
"""

import os
import torch
import logging

from torch import nn
from typing import Dict
from utils import main_utils
from datetime import datetime
from torch.optim import Optimizer
from utils.types import PathT, InputSample
from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    """
    Train logger class, covers all reports to tensorboard, console and files.
    """
    def __init__(self, exp_name_prefix: str = '', logs_dir: PathT = 'logs'):
        """
        :param exp_name_prefix: const prefix for specific experiment. For example, Random_exp_..
        :param logs_dir: directory for text file and tensorboard files
        """
        self.exp_name_prefix = exp_name_prefix

        main_utils.make_dir(logs_dir)

        self.exp_name = f'{exp_name_prefix}_{self._get_time_string()}'
        self.exp_dir = os.path.join(logs_dir, self.exp_name)

        main_utils.make_dir(self.exp_dir)

        # Init tensorboard
        tensorboard_path = os.path.join(logs_dir, 'tensorboard', self.exp_name)
        self.tensorboard_writer = SummaryWriter(tensorboard_path)

        # Init console and file logger
        self.logger = self._init_logger(self.exp_dir, self.exp_name)

    def write(self, text: str, epoch: int = None, severity: str = 'info') -> None:
        """
        Output log in severity of info or warning. Specify epoch if given
        :param text: text of the log
        :param epoch: if given, adds the current epoch.
        :param severity: info or warning are valid
        """
        log = ''

        if epoch is not None:
            log += f'(EPOCH {epoch}) '
        log += f'{text}'

        if severity == 'warning':
            self.logger.warning(log)
        else:
            self.logger.info(log)

    def report_metrics_hyper_params(self, hyper_params: Dict, metrics: Dict) -> None:
        """
        Adds hyper parameters and metrics to tensorboard
        :param hyper_params: dictionary holds all the hyper parameters {hyper_param: value}
        :param metrics: dictionary holds all the metrics {metric: value}
        """
        self.tensorboard_writer.add_hparams(hyper_params, metrics)

    def report_scalar(self, tag: str, scalar_value: float, step: int) -> None:
        """
        Report a scalar to tensorboard
        :param tag: report the scalar under tag
        :param scalar_value:
        :param step: epoch
        """
        self.tensorboard_writer.add_scalar(tag, scalar_value, step)

    def report_graph(self, model: nn.Module, model_input: InputSample) -> None:
        """
        Report a model structure to tensorboard
        :param model: a model instance
        :param model_input: tensor or list of tensors
        """
        self.tensorboard_writer.add_graph(model, model_input)

    def save_model(self, model: nn.Module, epoch: int, optimizer: Optimizer = None) -> None:
        """
        Saving a model
        :param model: a model instance with trained weights
        :param epoch: current epoch
        :param optimizer: (optional), add saves the optimizer state
        """
        model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict()
        }

        if optimizer is not None:
            model_dict['optimizer_state'] = optimizer.state_dict()

        temp_exp_dir = self.exp_dir.split("/") # ['logs', 'VQA_time']
        dir_path = os.path.join(temp_exp_dir[0], "saved_models", temp_exp_dir[1]) # str: logs/saved_models/VQA_time"
        main_utils.make_dir(dir_path)
        model_path = os.path.join(dir_path, 'model.pth')
        torch.save(model_dict, model_path)

    @staticmethod
    def _get_time_string() -> str:
        """
        :return: string with the current time in the format 'month_day_hour_minute_second'
        """
        time = datetime.now()

        return f'{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}'

    @staticmethod
    def _init_logger(exp_dir: PathT, exp_name: str) -> logging.Logger:
        """
        Create a logger instance
        :param exp_dir: the directory that will hold the log file
        :param exp_name: the experiment name
        :return: a logger
        """
        # Create logger
        _logger = logging.getLogger('TRAIN')
        _logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] - %(message)s')

        # Add console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        _logger.addHandler(ch)

        # Add file handler
        file_path = os.path.join(exp_dir, f'{exp_name}.log')
        fh = logging.FileHandler(filename=file_path)
        fh.setFormatter(formatter)
        _logger.addHandler(fh)

        return _logger

    def report_scalars(self, scalars, epoch):
        """
        Report batch of scalars
        :param scalars: {scalar_key: scalar_value}. For instance: {'Accuracy_train': 99.32}
        :param epoch:
        """
        for scalar, scalar_value in scalars.items():
            self.report_scalar(scalar, scalar_value, epoch)

    def write_epoch_statistics(self, epoch: int, epoch_time: float, train_loss: float, norm: float,
                               train_soft_acc: float, train_acc: float, eval_soft_acc: float, eval_acc: float) -> None:
        """

        :param epoch:
        :param epoch_time:
        :param train_loss:
        :param norm:
        :param train_soft_acc:
        :param train_acc:
        :param eval_soft_acc:
        :param eval_acc:
        :return:
        """
        text = 'Time: %.2f, secs' % epoch_time
        text += 'Gradient norm: %.4f, ' % norm
        text += 'Train loss: %.2f, ' % train_loss
        text += 'Train soft acc: %.2f, ' % train_soft_acc
        text += 'Train acc: %.4f, ' % train_acc
        text += 'Val soft acc: %.4f ' % eval_soft_acc
        text += 'Val acc: %.2f ' % eval_acc

        self.write(text, epoch)


if __name__ == '__main__':
    logger = TrainLogger(exp_name_prefix='test', logs_dir='test_dir')
    for i in range(15):
        logger.write('I am without epoch')
    for i in range(15):
        logger.write('I am with epoch', epoch=i)