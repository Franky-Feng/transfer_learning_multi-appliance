import os
import numpy as np
import Arguments

def preprocess_data(file_path, appliance_idx, save_path, aggregate_mean, aggregate_std, params_appliance,save_name):
    test_set = np.load(file_path)
    test_set_copy = test_set.copy()

    for j in range(len(test_set_copy)):
        test_set_copy[j, :, 0] = test_set_copy[j, :, 0] * aggregate_std + aggregate_mean
        for i, app in enumerate(appliance):
            test_set_copy[j, :, i + 1] = test_set_copy[j, :, i + 1] * params_appliance[str(app)]['std'] + \
                                         params_appliance[str(app)]['mean']

    count_values = [[] for _ in range(len(appliance))]
    for j in range(len(test_set_copy)):
        for i, app in enumerate(appliance):
            if max(test_set_copy[j, :, i + 1]) > params_appliance[str(app)]['on_power_threshold']:
                count_values[i].append(j)

    mean = []
    for i in range(len(appliance)):
        mae_mean = len(count_values[i])
        print('appliance:', appliance[i], 'sum:', mae_mean)
        mean.append(mae_mean)
    length = len(test_set_copy)
    array = np.arange(length)

    extracted_data = test_set[count_values[appliance_idx]]
    new = np.setdiff1d(array, count_values[appliance_idx])
    random_indices = np.random.choice(new, size=min(mean[appliance_idx],len(new)), replace=False)
    repeated_indices = test_set[random_indices]
    out_set = np.concatenate([extracted_data, repeated_indices], axis=0)

    np.save(os.path.join(save_path, save_name), out_set)
    print(len(out_set))

appliance = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
# change this to different appliance
app_num = 1
para = params_appliance = Arguments.refit_params_appliance
aggregate_mean = params_appliance['aggregate']['mean']
aggregate_std = params_appliance['aggregate']['std']

# 处理训练集
train_file_path = 'total/train_set2.npy'
train_save_path = appliance[app_num] + '/'
save_name = 'train_set.npy'
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)
preprocess_data(train_file_path, app_num, train_save_path, aggregate_mean, aggregate_std, params_appliance,save_name)

# 处理验证集
val_file_path = 'total/val_set2.npy'
val_save_path = appliance[app_num] + '/'
save_name = 'val_set.npy'
if not os.path.exists(val_save_path):
    os.makedirs(val_save_path)
preprocess_data(val_file_path, app_num, val_save_path, aggregate_mean, aggregate_std, params_appliance,save_name)


