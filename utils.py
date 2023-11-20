from datetime import datetime, date
import os
import logging
import logging.handlers
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import Arguments

appliance_name = Arguments.appliance_name


def get_user_input(args):
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    args.num_epochs = int(input('Input training epochs: '))


def acc_precision_recall_f1_score(pred, status):
    assert pred.shape == status.shape

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[
            0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / \
                   np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)


def f1_score_single(pred, status):
    assert pred.shape == status.shape

    pred = pred.reshape(-1)
    status = status.reshape(-1)
    accs, precisions, recalls, f1_scores = [], [], [], []

    tn, fp, fn, tp = confusion_matrix(status, pred, labels=[0, 1]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / np.max((tp + fp, 1e-9))
    recall = tp / np.max((tp + fn, 1e-9))
    f1_score = 2 * (precision * recall) / \
               np.max((precision + recall, 1e-9))

    accs.append(acc)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)


def relative_absolute_error(pred, label):
    assert pred.shape == label.shape

    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    temp = np.full(label.shape, 1e-9)
    relative, absolute, sum_err = [], [], []

    for i in range(label.shape[-1]):
        relative_error = np.mean(np.nan_to_num(np.abs(label[:, i] - pred[:, i]) / np.max(
            (label[:, i], pred[:, i], temp[:, i]), axis=0)))
        absolute_error = np.mean(np.abs(label[:, i] - pred[:, i]))

        relative.append(relative_error)
        absolute.append(absolute_error)

    return np.array(relative), np.array(absolute)


def relative_absolute_error_single(pred, label):
    assert pred.shape == label.shape

    pred = pred.reshape(-1)
    label = label.reshape(-1)
    temp = np.full(label.shape, 1e-9)
    relative, absolute, sum_err = [], [], []
    relative_error = np.mean(np.nan_to_num(np.abs(label - pred) / np.max(
        (label, pred, temp), axis=0)))

    absolute_error = np.mean(np.abs(label - pred))

    relative.append(relative_error)
    absolute.append(absolute_error)

    return np.array(relative), np.array(absolute)


def absolute_error(pred, label):
    assert pred.shape == label.shape

    pred = pred.reshape(-1)
    label = label.reshape(-1)
    absolute = []

    absolute_error = np.mean(np.abs(label - pred))
    absolute.append(absolute_error)

    return np.array(absolute)


def sae_error(pred, label):
    assert pred.shape == label.shape
    sae = []
    diff = 0
    if len(pred.shape) == 1:
        diff += abs(np.sum(label) - np.sum(pred))
        diff = diff /  pred.shape[0]
        sae.append(diff)
    else:
        N = pred.shape[1]
        for i in range(pred.shape[0]):
            diff += abs(np.sum(label[i]) - np.sum(pred[i]))
        diff = diff / (N * pred.shape[0])
        sae.append(diff)

    return np.array(sae)


def mkdir(dirName):
    if not os.path.exists(dirName):
        if os.name == 'nt':
            os.system('mkdir {}'.format(dirName.replace('/', '\\')))
        else:
            os.system('mkdir -p {}'.format(dirName))


def setup_log(subName='', tag='root'):
    # create logger
    logger = logging.getLogger(tag)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    # file log
    log_name = tag + datetime.now().strftime('log_%Y_%m_%d.log')

    log_path = os.path.join('log', subName, log_name)
    fh = logging.handlers.RotatingFileHandler(
        log_path, mode='a', maxBytes=100 * 1024 * 1024, backupCount=1, encoding='utf-8'
    )

    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


class EarlyStopping:
    def __init__(self, logger, patience=7, verbose=False, delta=0, best_score=None):
        self.patience = patience
        self.verbose = verbose  # true
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.early_stop = False

    def save_checkpoint(self, val_loss, net, path):
        if self.verbose:
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        saveModel(self.logger, net, path)
        self.val_loss_min = val_loss


def saveModel(logger, net, path, optimizer=None, path2=None):
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)

    torch.save(net, path)

    if optimizer is not None and path2 is not None:
        torch.save(optimizer, path2)

    logger.info(f'Model saved')


def tensor_cutoff_energy(data, max):
    data[data < 5] = 0
    min_value = 0.0
    max_value = float(max)
    data = torch.clamp(data, min=min_value, max=max_value)
    return data


def numpy_cutoff_energy(data, max):
    data[data < 5] = 0
    min_value = 0.0
    max_value = float(max)
    data = np.clip(data, a_min=min_value, a_max=max_value)
    return data


def numpy_compute_status(data,para):
    status = np.zeros(data.shape)
    columns = data.shape[-1]
    for i in range(columns):
        threshold = para[appliance_name[i]]['on_power_threshold']
        mean = para[appliance_name[i]]['mean']
        std = para[appliance_name[i]]['std']
        threshold_scale = (threshold - mean) / std
        status[:, :, i] = np.where(data[:, :, i] > threshold_scale, 1, 0)

    return status
