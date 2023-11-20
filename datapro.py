import random
import numpy as np
import torch
import torch.utils.data as data_utils
from abc import *
import utils

torch.set_default_tensor_type(torch.DoubleTensor)


class NILMDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.batch_size = args.batch_size
        self.train_dataset, self.val_dataset = dataset.get_datasets()

    def get_dataloaders(self):
        train_loader = self._get_loader(self.train_dataset)
        val_loader = self._get_loader(self.val_dataset)
        return train_loader, val_loader

    def test_dataloaders(self):
        val_loader = self._get_loader(self.val_dataset)
        return val_loader

    def _get_loader(self, dataset):
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class NILMDataset(data_utils.Dataset):
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s

    def __len__(self):
        return int(len(self.x))

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        s = self.s[index]
        return torch.tensor(x), torch.tensor(y), torch.tensor(s)


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, training_path, val_path, para):
        self.training_path = training_path
        self.val_path = val_path
        self.x_train, self.y_train, self.status_train = self.load_data(self.training_path, para)
        self.x_val, self.y_val, self.status_val = self.load_data(self.val_path, para)

    def load_data(self, path, para):
        test_set = np.load(path)
        status = utils.numpy_compute_status(test_set[:, :, 1:], para)
        return test_set[:, :, 0], test_set[:, :, 1:], status

    def get_datasets(self):
        val = NILMDataset(self.x_val, self.y_val, self.status_val)
        train = NILMDataset(self.x_train, self.y_train, self.status_train)
        return train, val


class AbstractDataset_appliance(metaclass=ABCMeta):
    def __init__(self, training_path, val_path, para,channel):
        self.training_path = training_path
        self.val_path = val_path
        self.x_train, self.y_train, self.status_train = self.load_data(self.training_path, para,channel)
        self.x_val, self.y_val, self.status_val = self.load_data(self.val_path, para,channel)

    def load_data(self, path, para,channel):
        test_set = np.load(path)
        status = utils.numpy_compute_status(test_set[:, :, 1:], para)
        return test_set[:, :, 0], test_set[:, :, channel+1], status[:, :, channel]

    def get_datasets(self):
        val = NILMDataset(self.x_val, self.y_val, self.status_val)
        train = NILMDataset(self.x_train, self.y_train, self.status_train)
        return train, val
