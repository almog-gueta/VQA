import sys
import time
import torch
import torch.nn as nn

from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams
from utils.train_logger import TrainLogger
from torch.autograd import Variable
from utils.plot_convergence_graphs import plot_convergences

def get_metrics(best_eval_soft_acc: float, eval_soft_acc: float, eval_acc: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_soft_acc:
    :param eval_soft_acc:
    :param eval_acc:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_soft_acc,
            'Metrics/LastSoftAccuracy': eval_soft_acc,
            'Metrics/LastAccuracy': eval_acc,
            'Metrics/LastLoss': train_loss}


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, train_params: TrainParams,
          logger: TrainLogger, model_name: str) -> Metrics:
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param val_loader:
    :param train_params:
    :param logger:
    :return:
    """
    best_eval_soft_acc = 0
    best_eval_acc = 0
    best_epoch = -1

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], [] #plot

    for epoch in tqdm(range(train_params.num_epochs), file=sys.stdout):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()

        for i, (v, q, a) in enumerate(train_loader):
            if i % 10000 == 0:
                start_batch_time = time.time()
            if torch.cuda.is_available():
                v = v.cuda() # [batch_size, 3, resize_h, resize_w]
                q = (q[0].cuda(), q[1]) # questions: [batch_size, 19], q_lens: [batch_size, 1]
                a = a.cuda() # [batch_size, num_of_ans]


            y_hat = model((v,q)) # [batch_size, num_of_ans] softmax
            #majority_label = a.max(dim=1)[1] #torch.max(a, 1)[1].data
            loss = criterion(y_hat,a) #majority_label

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            # metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            # metrics['count_norm'] += 1

            # NOTE! This function compute scores correctly only for one hot encoding representation of the logits
            # soft accuracy
            batch_soft_acc = train_utils.compute_soft_accuracy_with_logits(y_hat, a.data).sum()
            metrics['train_soft_acc'] += batch_soft_acc.item()
            # accuracy
            batch_acc =  train_utils.compute_accuracy_with_logits(y_hat, a.data).sum()
            metrics['train_acc'] += batch_acc.item()

            metrics['train_loss'] += float(loss.item() * v.size(0)) # loss * batch_size

            # Report model to tensorboard
            # if epoch == 0 and i == 0:
            #     logger.report_graph(model, [v,q])

            if i % 10000 == 0:
                logger.write(f'done {i} batches, batch time took: {(time.time() - start_batch_time)/60} mins')
            # if i > 1000:
            #     break # todo

        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)

        metrics['train_soft_acc'] /= len(train_loader.dataset)
        metrics['train_soft_acc'] *= 100

        metrics['train_acc'] /= len(train_loader.dataset)
        metrics['train_acc'] *= 100

        # norm = metrics['total_norm'] / metrics['count_norm']

        logger.write(f'starting eval for epoch: {epoch + 1}')
        model.train(False)
        metrics['eval_soft_acc'], metrics['eval_acc'], metrics['eval_loss'] = evaluate(model, val_loader, criterion)
        model.train(True)

        # print(metrics)
        logger.write(metrics) # todo - delete
        # epoch_time = time.time() - t
        # logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
        #                               metrics['train_soft_acc'], metrics['train_acc'],
        #                               metrics['eval_soft_acc'], metrics['eval_acc'])
        #
        # scalars = {'Soft Accuracy/Train': metrics['train_soft_acc'],
        #            'Soft Accuracy/Validation': metrics['eval_soft_acc'],
        #            'Accuracy/Train': metrics['train_acc'],
        #            'Accuracy/Validation': metrics['eval_acc'],
        #            'Loss/Train': metrics['train_loss'],
        #            'Loss/Validation': metrics['eval_loss']}
        #
        # logger.report_scalars(scalars, epoch)
        #
        if metrics['eval_soft_acc'] > best_eval_soft_acc:
            best_eval_soft_acc = metrics['eval_soft_acc']
            best_epoch = epoch+1

        # torch.save(model.state_dict(), f"/home/student/hw2/logs/saved_models/{model_name}_model_dict_epoch_{epoch}.pth")

    # save trained CNN model
        #model_dict = model.state_dict()
        #model_dict['optimizer_state'] = optimizer.state_dict()
        #torch.save(model_dict, f"/home/student/hw2/logs/saved_models/trained_vqa_{epoch}.pth")

        #logger.save_model(model, epoch, optimizer)

        train_acc_list.append(metrics["train_soft_acc"]) #plot
        val_acc_list.append(metrics["eval_soft_acc"]) #plot
        train_loss_list.append(metrics["train_loss"]) #plot
        val_loss_list.append(metrics["eval_loss"]) #plot

        logger.write(f'done epoch {epoch+1} in {(time.time()-t) / 60} mins')

    print(f'best eval soft acc: {best_eval_soft_acc}, received in epoch: {best_epoch}')
    plot_convergences(train_acc_list, val_acc_list, train_loss_list, val_loss_list) #plot
    return get_metrics(best_eval_soft_acc, metrics['eval_soft_acc'], metrics['eval_acc'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.functional) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :param criterion: loss function
    :return: tuple of (soft accuracy, accuracy, loss) values
    """
    soft_accuracy = 0
    accuracy = 0
    loss = 0

    with torch.no_grad():
        for i, (v, q, a) in enumerate(dataloader):
            if torch.cuda.is_available():
                v = v.cuda()  # [batch_size, 3, resize_h, resize_w]
                q = (q[0].cuda(), q[1]) # questions: [batch_size, 19], q_lens: [batch_size, 1]
                a = a.cuda()  # [batch_size, num_of_ans]

            y_hat = model((v,q)) # [batch_size, num_of_ans] softmax

            # majority_label = torch.max(a, 1)[1].data
            loss += float(criterion(y_hat, a).item() * v.size(0))

            soft_accuracy += train_utils.compute_soft_accuracy_with_logits(y_hat, a).sum().item()
            accuracy += train_utils.compute_accuracy_with_logits(y_hat, a).sum().item()


    loss /= len(dataloader.dataset)
    soft_accuracy /= len(dataloader.dataset)
    soft_accuracy *= 100
    accuracy /= len(dataloader.dataset)
    accuracy *= 100

    return soft_accuracy, accuracy, loss