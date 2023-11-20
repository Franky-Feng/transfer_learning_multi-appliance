import torch
import torch.nn as nn
import torch.optim as optim
import utils
from datapro import AbstractDataset, NILMDataloader
from utils import *
import matplotlib.pyplot as plt
import Arguments
from tqdm import tqdm

channel = 4
appliance = Arguments.appliance_name
imgpath = '../result/' + 'refit/' + str(appliance[channel])
if not os.path.exists(imgpath):
    os.makedirs(imgpath)
para = Arguments.default_params_appliance
args = Arguments.args
torch.set_default_tensor_type(torch.DoubleTensor)
training_path = '../dataset_management/refit/total/train_set2.npy'
test_path = 'C:/Users/Administrator/Desktop/NILM6/数据备份/dataset_management/refit/washingmachine/val_set_small.npy'

print('val dataset: ' + test_path)
dataset = AbstractDataset(training_path, test_path, para)
dataloader = NILMDataloader(args, dataset)
train_loader, test_loader = dataloader.get_dataloaders()
folder_name = '-'.join('total')
export_root = 'C:/Users/Administrator/Desktop/NILM6/备份模型1009/experiments/t-o-t-a-l'

try:
    combined_model = torch.load(os.path.join(export_root, 'best_acc_model.pth'), map_location='cpu')
    # combined_model = torch.load(os.path.join(export_root, 'checkpoint_49.pth'), map_location='cpu')

    print('Successfully loaded previous model, continue training...')
except FileNotFoundError:
    print('Failed to load old model, continue training new model...')
    exit()

device = args.device
combined_model = combined_model.to(device)
combined_model.eval()
loss_values, relative_errors, absolute_errors = [], [], []
acc_values, precision_values, recall_values, f1_values, = [], [], [], []

mse = nn.MSELoss()
count = 0
with torch.no_grad():
    tqdm_dataloader = tqdm(test_loader)
    for batch_idx, batch in enumerate(tqdm_dataloader):
        seqs, labels_energy, status = batch
        seqs, labels_energy = seqs.to(device), labels_energy.to(device)
        combined_tensor, y_pred_status = combined_model(seqs)  # 128 480 5
        seqs = seqs * para['aggregate']['std'] + para['aggregate']['mean']
        for i in range(len(appliance)):
            labels_energy[:, :, i] = labels_energy[:, :, i] * para[str(appliance[i])]['std'] + \
                                     para[str(appliance[i])]['mean']
            combined_tensor[:, :, i] = combined_tensor[:, :, i] * para[str(appliance[i])]['std'] + \
                                       para[str(appliance[i])]['mean']
            labels_energy[:, :, i] = utils.tensor_cutoff_energy(labels_energy[:, :, i],
                                                                para[str(appliance[i])]['max_on_power'])
            combined_tensor[:, :, i] = utils.tensor_cutoff_energy(combined_tensor[:, :, i],
                                                                  para[str(appliance[i])]['max_on_power'])

        mse_loss = mse(combined_tensor.contiguous().view(-1).double(), labels_energy.contiguous().view(-1).double())
        total_loss = mse_loss

        tqdm_dataloader.set_description('Validation, err {:.2f}'.format(total_loss))
        loss_values.append(total_loss.item())

        for j in range(labels_energy.shape[0]):  # 32
            logitsss_gpu = combined_tensor.cuda()
            logitsss_cpu = logitsss_gpu.cpu()
            seq = logitsss_cpu[j].numpy()

            labels_energy_gpu = labels_energy.cuda()
            labels_energy_cpu = labels_energy_gpu.cpu()
            energy_label = labels_energy_cpu[j].numpy()

            aggregate_gpu = seqs.cuda()
            aggregate_cpu = aggregate_gpu.cpu()
            agg = aggregate_cpu[j].numpy()

            status_gpu = status.cuda()
            status_cpu = status_gpu.cpu()
            status1 = status_cpu[j].numpy()

            y_pred_status = y_pred_status.cuda()
            y_pred_status = y_pred_status.cpu()
            y_pred_status1 = y_pred_status[j].numpy()

            # has_non_zero_element = np.any(energy_label != 0)
            has_non_zero_element = np.any(
                energy_label[:, channel] > para[str(appliance[channel])]['on_power_threshold'])
            if has_non_zero_element != 0:
                if (count == 0) | (count == 4):
                    count = 0
                    fig, axs = plt.subplots(1, 4, figsize=(20, 3))
                if count < 4:
                    axs[count].plot(agg, label='aggregate', linestyle='--', color='gray')
                    axs[count].plot(energy_label[:, channel], label='Ground_truth', color='blue')
                    axs[count].plot(seq[:, channel], label='Disaggregation', color='red')
                    # axs[i].plot(status1[:, i], label='True status', color='black')
                    # axs[i].plot(y_pred_status1[:, i], label='predict status', color='yellow')
                    axs[count].legend(fontsize='small')
                    axs[count].legend()
                    count += 1

                    if count == 4:
                        plt.tight_layout()
                        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
                        file_name = 'result_plot_' + current_time + '.png'
                        plt.savefig(os.path.join(imgpath, file_name))
                        # plt.show()
                        plt.close()
return_rel_err = np.array(loss_values).mean(axis=0)
print(return_rel_err)
