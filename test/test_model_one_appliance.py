import torch
import torch.nn as nn
import torch.optim as optim
import utils
from datapro import AbstractDataset, NILMDataloader, AbstractDataset_appliance
from utils import *
import matplotlib.pyplot as plt
import Arguments
from tqdm import tqdm

para = Arguments.redd_params_appliance
appliance = Arguments.appliance_name
args = Arguments.args
torch.set_default_tensor_type(torch.DoubleTensor)
test_path = args.test_dataset_path
training_path = args.training_dataset_path
channel = args.channel

print('test dataset: ' + test_path)
# Build a dataset
dataset = AbstractDataset_appliance(training_path, test_path, para, channel)
dataloader = NILMDataloader(args, dataset)
train_loader, test_loader = dataloader.get_dataloaders()

export_root = args.model_path

try:
    combined_model = torch.load(export_root, map_location='cpu')
    print('Successfully loaded previous model...')
except FileNotFoundError:
    print('Failed to load old model...')
    exit()

device = args.device
combined_model = combined_model.to(device)
combined_model.eval()
absolute_errors = []
signal_aggregate_error = []
f1_values = []
mse = nn.MSELoss()
with torch.no_grad():
    tqdm_dataloader = tqdm(test_loader)
    for batch_idx, batch in enumerate(tqdm_dataloader):
        seqs, labels_energy, status = batch
        if status.shape[0] == 1:
            continue
        seqs, labels_energy = seqs.to(device), labels_energy.to(device)
        combined_tensor, y_pred_status = combined_model(seqs)  # 128 480 5
        threshold = 0.5
        y_pred_binary = (y_pred_status >= threshold).to(torch.int)
        seqs = seqs * para['aggregate']['std'] + para['aggregate']['mean']

        labels_energy[:, :] = labels_energy[:, :] * para[str(appliance[channel])]['std'] + \
                                 para[str(appliance[channel])]['mean']
        combined_tensor[:, :] = combined_tensor[:, :] * para[str(appliance[channel])]['std'] + \
                                   para[str(appliance[channel])]['mean']
        labels_energy[:, :] = utils.tensor_cutoff_energy(labels_energy[:, :],
                                                            para[str(appliance[channel])]['max_on_power'])
        combined_tensor[:, :] = utils.tensor_cutoff_energy(combined_tensor[:, :],
                                                              para[str(appliance[channel])]['max_on_power'])

        mae = absolute_error(labels_energy[:, :].detach().cpu().numpy().squeeze(),
                             combined_tensor[:, :].detach().cpu().numpy().squeeze())
        absolute_errors.append(mae)
        sae = sae_error(labels_energy[:, :].detach().cpu().numpy().squeeze(),
                        combined_tensor[:, :].detach().cpu().numpy().squeeze())
        signal_aggregate_error.append(sae)

        status_gpu = status.cuda()
        status_cpu = status_gpu.cpu()
        status1 = status_cpu.numpy()
        if 'status_list' not in locals():
            status_list = status1
        else:
            status_list = np.vstack([status_list, status1])

        y_pred_status = y_pred_binary.cuda()
        y_pred_status = y_pred_status.cpu()
        y_pred_status1 = y_pred_status.numpy()
        if 'y_pred_binary_list' not in locals():
            y_pred_binary_list = y_pred_status1
        else:
            y_pred_binary_list = np.vstack([y_pred_binary_list, y_pred_status1])

# Calculate the total data mae, sae
mae_mean = np.mean(absolute_errors)
sae_mean = np.mean(signal_aggregate_error)
print('appliance:', appliance[channel], 'mae:', mae_mean, 'sae:', sae_mean)

# Calculate f1
acc, precision, recall, f1 = f1_score_single(y_pred_binary_list, status_list)
print("""""""""""""")
print('f1:', f1)
print('acc:', acc, 'precision:', precision, 'recall:', recall)
