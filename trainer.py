import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.autograd.gradcheck import zero_gradients
from tqdm import tqdm

import os
import json
import random
import numpy as np
from abc import *
from pathlib import Path

import utils
from utils import *
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)


class Trainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, export_root, optimizer=None):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.to(self.device)
        self.export_root = Path(export_root)

        self.normalize = args.normalize
        self.denom = args.denom

        self.train_loader = train_loader
        self.val_loader = val_loader
        if optimizer == None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def train(self, logger):
        val_rel_err, val_abs_err = [], []
        val_acc, val_precision, val_recall, val_f1 = [], [], [], []
        loss_re = []
        early_stopping_all = utils.EarlyStopping(logger, patience=60, verbose=True)
        logger.info(f"Validation: ")
        best_loss, best_average = self.validate()
        logger.info(f"loss: {best_loss},MAE: {best_average[1]},rel_err:{best_average[0]},"
                    f"acc:{best_average[2]},f1:{best_average[5]}")
        best_score = -best_average[5] + best_average[1]
        for epoch in range(self.num_epochs):
            checkpointName = self.export_root.joinpath("checkpoint_" + str(epoch) + '.pth')
            checkpointName2 = self.export_root.joinpath("checkpoint_" + str(epoch) + 'optimizer' + '.pth')
            utils.saveModel(logger, self.model, checkpointName, self.optimizer, checkpointName2)

            logger.info(f"# of epoches: {epoch}")
            self.train_one_epoch(epoch + 1, logger)

            loss, average = self.validate()
            score = -average[5] + average[1]
            if score < best_score:
                best_score = score
                best_loss = loss
                best_average = average
            logger.info(f"Validation: ")
            logger.info(
                f"best_loss: {best_loss},best_MAE: {best_average[1]},best_rel_err:{best_average[0]},best_acc:{best_average[2]},best_f1:{best_average[5]}")
            logger.info(f"loss: {loss},MAE: {average[1]},rel_err:{average[0]},acc:{average[2]},f1:{average[5]}")

            loss_re.append(loss.mean())
            val_rel_err.append(average[0])
            val_abs_err.append(average[1])
            val_acc.append(average[2])
            val_precision.append(average[3])
            val_recall.append(average[4])
            val_f1.append(average[5])
            if epoch % 10 == 0:
                checkpointName = self.export_root.joinpath("checkpoint_" + str(epoch) + '.pth')
                checkpointName2 = self.export_root.joinpath("checkpoint_" + str(epoch) + 'optimizer' + '.pth')
                utils.saveModel(logger, self.model, checkpointName, self.optimizer, checkpointName2)

            early_stopping_all(score, self.model, self.export_root.joinpath('best_acc_model.pth'))
            if early_stopping_all.early_stop:
                print("Early stopping")
                break

        logger.info(f"summary: ")
        logger.info(f"loss: {loss_re}")
        logger.info(f"val_rel_err: {val_rel_err}")
        logger.info(f"val_abs_err: {val_abs_err}")
        logger.info(f"val_acc: {val_acc}")
        logger.info(f"val_precision: {val_precision}")
        logger.info(f"val_recall: {val_recall}")
        logger.info(f"val_f1: {val_f1}")

    def train_one_epoch(self, epoch, logger):
        global average_loss
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(
                self.device)
            self.optimizer.zero_grad(set_to_none=True)
            y_pred, y_pred_status = self.model(seqs)
            loss_r = self.mse(y_pred, labels_energy)
            loss_c = self.bce(y_pred_status, status)
            total_loss = loss_r + loss_c
            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        logger.info(f"Train Epoch: {epoch}")
        logger.info(f"loss: {average_loss}")

    def validate(self):
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values, = [], [], [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(
                    self.device)
                self.optimizer.zero_grad()
                y_pred, y_pred_status = self.model(seqs)
                loss_r = self.mse(y_pred, labels_energy)
                loss_c = self.bce(y_pred_status, status)
                total_loss = loss_r + loss_c
                threshold = 0.5
                y_pred_binary = (y_pred_status >= threshold).to(torch.int)
                rel_err, abs_err = relative_absolute_error(y_pred.detach().cpu().numpy().squeeze(),
                                                           labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())
                acc, precision, recall, f1 = acc_precision_recall_f1_score(y_pred_binary.detach(
                ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())
                loss_values.append(total_loss.item())
                tqdm_dataloader.set_description('Validation, err {:.2f}'.format(total_loss))
        return_rel_err = np.array(loss_values).mean(axis=0)
        return_rel_err2 = np.array(relative_errors).mean(axis=0)
        return_abs_err = np.array(absolute_errors).mean(axis=0)
        return_acc = np.array(acc_values).mean(axis=0)
        return_precision = np.array(precision_values).mean(axis=0)
        return_recall = np.array(recall_values).mean(axis=0)
        return_f1 = np.array(f1_values).mean(axis=0)
        average = []
        average.append(np.mean(return_rel_err2))
        average.append(np.mean(return_abs_err))
        average.append(np.mean(return_acc))
        average.append(np.mean(return_precision))
        average.append(np.mean(return_recall))
        average.append(np.mean(return_f1))
        return return_rel_err, average

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr,
                             momentum=args.momentum)
        else:
            raise ValueError

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(
                self.export_root.joinpath('best_acc_model.pth')))
            self.model.to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model, self.export_root.joinpath('best_acc_model.pth'))

    def _save_values(self, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        torch.save(self.model, self.export_root.joinpath('best_acc_model.pth'))

    def _save_result(self, data, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)


class Plot_input(metaclass=ABCMeta):
    def __init__(self, args, train_loader, val_loader):
        self.args = args
        self.device = args.device
        self.train_loader = train_loader
        self.val_loader = val_loader

    def start_plot(self):
        tqdm_dataloader = tqdm(self.train_loader)
        flag = 1
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            for i in range(seqs.shape[0]):
                seq = seqs[i].numpy()
                energy_label = labels_energy[i].numpy()
                has_non_zero_element = np.any(energy_label != 0)
                if has_non_zero_element != 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(seq, label='Sequence Data', color='blue')
                    plt.plot(energy_label, label='Energy Labels', color='red')
                    plt.legend()
                    plt.show()

                # flag = int(input('input 0 will stop plt:'))
                # if flag == 0:
                #     break
                #
                # plt.figure(figsize=(10, 5))
                # plt.plot(seq, label='Sequence Data', color='blue')
                # plt.plot(energy_label, label='Energy Labels', color='red')
                # plt.legend()
                #
                # plt.show()


class Trainer_single(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, export_root, channel):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.to(self.device)
        self.export_root = Path(export_root)
        self.channel = channel
        self.normalize = args.normalize
        self.denom = args.denom
        # if self.normalize == 'mean':
        #     self.x_mean, self.x_std = stats
        #     self.x_mean = torch.tensor(self.x_mean).to(self.device)
        #     self.x_std = torch.tensor(self.x_std).to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        # tqdm_dataloader = tqdm(self.train_loader)
        # for batch_idx, batch in enumerate(tqdm_dataloader):
        #     seqs, labels_energy, status = batch
        #     seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(
        #         self.device)
        #     with torch.no_grad():
        #         _ = self.model.new_model3(seqs)
        #         _ = self.model.new_model(seqs)
        #         _ = self.model.new_model4(seqs)
        #         _ = self.model.new_model5(seqs)
        #     break

    def train(self, logger):
        val_rel_err, val_abs_err = [], []
        val_acc, val_precision, val_recall, val_f1 = [], [], [], []
        loss_re = []
        early_stopping_all = utils.EarlyStopping(logger, patience=100, verbose=True)
        logger.info(f"Validation: ")
        best_loss, best_average = self.validate()
        logger.info(f"loss: {best_loss},MAE: {best_average[1]},rel_err:{best_average[0]},"
                    f"acc:{best_average[2]},f1:{best_average[5]}")
        best_score = -best_average[5] + best_average[1]
        for epoch in range(self.num_epochs):
            logger.info(f"# of epoches: {epoch}")
            self.train_one_epoch(epoch + 1, logger)
            loss, average = self.validate()
            score = -average[5] + average[1]

            if score < best_score:
                best_score = score
                best_loss = loss
                best_average = average
            logger.info(f"Validation: ")
            logger.info(
                f"best_loss: {best_loss},best_MAE: {best_average[1]},best_rel_err:{best_average[0]},best_acc:{best_average[2]},best_f1:{best_average[5]}")
            logger.info(f"loss: {loss},MAE: {average[1]},rel_err:{average[0]},acc:{average[2]},f1:{average[5]}")

            loss_re.append(loss.mean())
            val_rel_err.append(average[0])
            val_abs_err.append(average[1])
            val_acc.append(average[2])
            val_precision.append(average[3])
            val_recall.append(average[4])
            val_f1.append(average[5])
            if epoch % 10 == 0:
                checkpointName = self.export_root.joinpath("checkpoint_" + str(epoch) + '.pth')
                utils.saveModel(logger, self.model, checkpointName)
            early_stopping_all(score, self.model, self.export_root.joinpath('best_acc_model.pth'))
            if early_stopping_all.early_stop:
                print("Early stopping")
                break

        logger.info(f"summary: ")
        logger.info(f"loss: {loss_re}")
        logger.info(f"val_rel_err: {val_rel_err}")
        logger.info(f"val_abs_err: {val_abs_err}")
        logger.info(f"val_acc: {val_acc}")
        logger.info(f"val_precision: {val_precision}")
        logger.info(f"val_recall: {val_recall}")
        logger.info(f"val_f1: {val_f1}")

    def train_one_epoch(self, epoch, logger):
        global average_loss
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(
                self.device)
            self.optimizer.zero_grad(set_to_none=True)
            y_pred, y_pred_status = self.model(seqs)
            y_pred = y_pred[:, :, self.channel]
            labels_energy = labels_energy[:, :, self.channel]
            y_pred_status = y_pred_status[:, :, self.channel]
            status = status[:, :, self.channel]
            loss_r = self.mse(y_pred, labels_energy)
            loss_c = self.bce(y_pred_status, status)
            total_loss = loss_r + loss_c
            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        logger.info(f"Train Epoch: {epoch}")
        logger.info(f"loss: {average_loss}")

    def validate(self):
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values, = [], [], [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(
                    self.device)
                self.optimizer.zero_grad()
                y_pred, y_pred_status = self.model(seqs)
                y_pred = y_pred[:, :, self.channel]
                labels_energy = labels_energy[:, :, self.channel]
                y_pred_status = y_pred_status[:, :, self.channel]
                status = status[:, :, self.channel]
                loss_r = self.mse(y_pred, labels_energy)
                loss_c = self.bce(y_pred_status, status)
                total_loss = loss_r + loss_c
                threshold = 0.5
                y_pred_binary = (y_pred_status >= threshold).to(torch.int)
                rel_err, abs_err = relative_absolute_error_single(y_pred.detach().cpu().numpy().squeeze(),
                                                                  labels_energy.detach().cpu().numpy().squeeze())

                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())
                acc, precision, recall, f1 = f1_score_single(y_pred_binary.detach().cpu().numpy().squeeze(),
                                                             status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                loss_values.append(total_loss.item())
                tqdm_dataloader.set_description('Validation, err {:.2f}'.format(total_loss))
        return_loss = np.array(loss_values).mean(axis=0)
        return_rel_err2 = np.array(relative_errors).mean(axis=0)
        return_abs_err = np.array(absolute_errors).mean(axis=0)
        return_acc = np.array(acc_values).mean(axis=0)
        return_precision = np.array(precision_values).mean(axis=0)
        return_recall = np.array(recall_values).mean(axis=0)
        return_f1 = np.array(f1_values).mean(axis=0)
        average = []
        average.append(np.mean(return_rel_err2))
        average.append(np.mean(return_abs_err))
        average.append(np.mean(return_acc))
        average.append(np.mean(return_precision))
        average.append(np.mean(return_recall))
        average.append(np.mean(return_f1))
        return return_loss, average

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr,
                             momentum=args.momentum)
        else:
            raise ValueError

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(
                self.export_root.joinpath('best_acc_model.pth')))
            self.model.to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model, self.export_root.joinpath('best_acc_model.pth'))

    def _save_values(self, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        torch.save(self.model, self.export_root.joinpath('best_acc_model.pth'))

    def _save_result(self, data, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)


class Trainer_one_appliance(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, export_root, optimizer=None):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.to(self.device)
        self.export_root = Path(export_root)

        self.normalize = args.normalize
        self.denom = args.denom

        self.train_loader = train_loader
        self.val_loader = val_loader
        if optimizer == None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def train(self, logger):
        val_rel_err, val_abs_err = [], []
        val_acc, val_precision, val_recall, val_f1 = [], [], [], []
        loss_re = []
        early_stopping_all = utils.EarlyStopping(logger, patience=60, verbose=True)
        logger.info(f"Validation: ")
        best_loss, best_average = self.validate()
        logger.info(f"loss: {best_loss},MAE: {best_average[1]},rel_err:{best_average[0]},"
                    f"acc:{best_average[2]},f1:{best_average[5]}")
        best_score = -best_average[5] + best_average[1]
        for epoch in range(self.num_epochs):
            # checkpointName = self.export_root.joinpath("checkpoint_" + str(epoch) + '.pth')
            # checkpointName2 = self.export_root.joinpath("checkpoint_" + str(epoch) + 'optimizer' + '.pth')
            # utils.saveModel(logger, self.model, checkpointName, self.optimizer, checkpointName2)

            logger.info(f"# of epoches: {epoch}")
            self.train_one_epoch(epoch + 1, logger)

            loss, average = self.validate()
            score = -average[5] + average[1]
            if score < best_score:
                best_score = score
                best_loss = loss
                best_average = average
            logger.info(f"Validation: ")
            logger.info(
                f"best_loss: {best_loss},best_MAE: {best_average[1]},best_rel_err:{best_average[0]},best_acc:{best_average[2]},best_f1:{best_average[5]}")
            logger.info(f"loss: {loss},MAE: {average[1]},rel_err:{average[0]},acc:{average[2]},f1:{average[5]}")

            loss_re.append(loss.mean())
            val_rel_err.append(average[0])
            val_abs_err.append(average[1])
            val_acc.append(average[2])
            val_precision.append(average[3])
            val_recall.append(average[4])
            val_f1.append(average[5])
            if epoch % 10 == 0:
                checkpointName = self.export_root.joinpath("checkpoint_" + str(epoch) + '.pth')
                checkpointName2 = self.export_root.joinpath("checkpoint_" + str(epoch) + 'optimizer' + '.pth')
                utils.saveModel(logger, self.model, checkpointName, self.optimizer, checkpointName2)

            early_stopping_all(score, self.model, self.export_root.joinpath('best_acc_model.pth'))
            if early_stopping_all.early_stop:
                print("Early stopping")
                break

        logger.info(f"summary: ")
        logger.info(f"loss: {loss_re}")
        logger.info(f"val_rel_err: {val_rel_err}")
        logger.info(f"val_abs_err: {val_abs_err}")
        logger.info(f"val_acc: {val_acc}")
        logger.info(f"val_precision: {val_precision}")
        logger.info(f"val_recall: {val_recall}")
        logger.info(f"val_f1: {val_f1}")

    def train_one_epoch(self, epoch, logger):
        global average_loss
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            if status.shape[0] == 1:
                continue
            seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(
                self.device)
            self.optimizer.zero_grad(set_to_none=True)

            y_pred, y_pred_status = self.model(seqs)
            if y_pred_status.shape[0] == 1:
                y_pred_status = y_pred_status.squeeze()
            loss_r = self.mse(y_pred, labels_energy)
            loss_c = self.bce(y_pred_status, status)
            total_loss = loss_r + loss_c
            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())
            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        logger.info(f"Train Epoch: {epoch}")
        logger.info(f"loss: {average_loss}")

    def validate(self):
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values, = [], [], [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(
                    self.device)
                self.optimizer.zero_grad()
                y_pred, y_pred_status = self.model(seqs)
                if y_pred_status.shape != status.shape:
                    pass
                else:
                    loss_r = self.mse(y_pred, labels_energy)
                    loss_c = self.bce(y_pred_status, status)
                total_loss = loss_r + loss_c
                threshold = 0.5
                y_pred_binary = (y_pred_status >= threshold).to(torch.int)
                rel_err, abs_err = relative_absolute_error_single(y_pred.detach().cpu().numpy().squeeze(),
                                                                  labels_energy.detach().cpu().numpy().squeeze())

                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())
                acc, precision, recall, f1 = f1_score_single(y_pred_binary.detach(
                ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                loss_values.append(total_loss.item())
                tqdm_dataloader.set_description('Validation, err {:.2f}'.format(total_loss))
        return_rel_err = np.array(loss_values).mean(axis=0)
        return_rel_err2 = np.array(relative_errors).mean(axis=0)
        return_abs_err = np.array(absolute_errors).mean(axis=0)
        return_acc = np.array(acc_values).mean(axis=0)
        return_precision = np.array(precision_values).mean(axis=0)
        return_recall = np.array(recall_values).mean(axis=0)
        return_f1 = np.array(f1_values).mean(axis=0)
        average = []
        average.append(np.mean(return_rel_err2))
        average.append(np.mean(return_abs_err))
        average.append(np.mean(return_acc))
        average.append(np.mean(return_precision))
        average.append(np.mean(return_recall))
        average.append(np.mean(return_f1))
        return return_rel_err, average

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr,
                             momentum=args.momentum)
        else:
            raise ValueError

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(
                self.export_root.joinpath('best_acc_model.pth')))
            self.model.to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model, self.export_root.joinpath('best_acc_model.pth'))

    def _save_values(self, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        torch.save(self.model, self.export_root.joinpath('best_acc_model.pth'))

    def _save_result(self, data, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)
