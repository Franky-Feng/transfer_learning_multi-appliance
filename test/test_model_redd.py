import torch
import torch.nn as nn
import torch.optim as optim
import utils
from datapro import AbstractDataset, NILMDataloader
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

print('test dataset: ' + test_path)
# Build a dataset
dataset = AbstractDataset(training_path, test_path, para)
dataloader = NILMDataloader(args, dataset)
train_loader, test_loader = dataloader.get_dataloaders()

# folder_name = '-'.join('redd')
# export_root = '../experiments/' + folder_name
export_root = args.model_path

try:
    combined_model = torch.load(os.path.join(export_root, 'best_acc_model.pth'), map_location='cpu')
    # combined_model = torch.load(os.path.join(export_root, 'checkpoint_39.pth'), map_location='cpu')

    print('Successfully loaded previous model...')
except FileNotFoundError:
    print('Failed to load old model...')
    exit()

device = args.device
combined_model = combined_model.to(device)
combined_model.eval()
# acc_values, precision_values, recall_values, f1_values, = [], [], [], []
absolute_errors = [[] for _ in range(len(appliance))]
signal_aggregate_error = [[] for _ in range(len(appliance))]
f1_values = [[] for _ in range(len(appliance))]
mse = nn.MSELoss()
# Model initialization
with torch.no_grad():
    tqdm_dataloader = tqdm(test_loader)
    for batch_idx, batch in enumerate(tqdm_dataloader):
        seqs, labels_energy, status = batch
        seqs, labels_energy = seqs.to(device), labels_energy.to(device)
        combined_tensor, y_pred_status = combined_model(seqs)  # 128 480 5
        # combined_tensor = torch.cat([y, y2, y3, y4, y5], dim=2)
        threshold = 0.5
        y_pred_binary = (y_pred_status >= threshold).to(torch.int)
        seqs = seqs * para['aggregate']['std'] + para['aggregate']['mean']
        for i in range(len(appliance)-1):
            labels_energy[:, :, i+1] = labels_energy[:, :, i+1] * para[str(appliance[i+1])]['std'] + \
                                     para[str(appliance[i+1])]['mean']
            combined_tensor[:, :, i] = combined_tensor[:, :, i] * para[str(appliance[i+1])]['std'] + \
                                       para[str(appliance[i+1])]['mean']
            labels_energy[:, :, i+1] = utils.tensor_cutoff_energy(labels_energy[:, :, i+1],
                                                                para[str(appliance[i+1])]['max_on_power'])
            combined_tensor[:, :, i] = utils.tensor_cutoff_energy(combined_tensor[:, :, i],
                                                                  para[str(appliance[i+1])]['max_on_power'])

        labels_energy = labels_energy[:, :, 1:]
        status = status[:, :, 1:]

        for i in range(len(appliance)-1):
            mae = absolute_error(labels_energy[:, :, i].detach().cpu().numpy().squeeze(),
                                 combined_tensor[:, :, i].detach().cpu().numpy().squeeze())
            absolute_errors[i].append(mae)
            sae = sae_error(labels_energy[:, :, i].detach().cpu().numpy().squeeze(),
                            combined_tensor[:, :, i].detach().cpu().numpy().squeeze())
            signal_aggregate_error[i].append(sae)

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
for i in range(len(appliance)-1):
    mae_mean = np.mean(absolute_errors[i])
    sae_mean = np.mean(signal_aggregate_error[i])
    print('appliance:', appliance[i+1], 'mae:', mae_mean, 'sae:', sae_mean)

# Calculate f1
acc, precision, recall, f1 = acc_precision_recall_f1_score(y_pred_binary_list, status_list)
print("""""""""""""")
print('f1:', f1)
print('acc:', acc, 'precision:', precision, 'recall:', recall)


