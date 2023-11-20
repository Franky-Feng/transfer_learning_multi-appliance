import numpy as np
import matplotlib.pyplot as plt
import Arguments

appliance = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
para = params_appliance = Arguments.ukdale_params_appliance
aggregate_mean = params_appliance['aggregate']['mean']
aggregate_std = params_appliance['aggregate']['std']
save_path = 'total/'

file_path = 'total/test_set.npy'
test_set = np.load(file_path)
test_set_copy = test_set.copy()

print(len(test_set_copy))
for j in range(len(test_set_copy)):
    test_set_copy[j, :, 0] = test_set_copy[j, :, 0] * aggregate_std + aggregate_mean
    for i in range(1, len(appliance) + 1):
        test_set_copy[j, :, i] = test_set_copy[j, :, i] * params_appliance[str(appliance[i - 1])]['std'] + \
                                 params_appliance[str(appliance[i - 1])]['mean']

# 计算每个电器数据的数目
count_values = [[] for _ in range(len(appliance))]
for j in range(len(test_set_copy)):
    for i in range(1, len(appliance) + 1):
        if max(test_set_copy[j, :, i]) > params_appliance[str(appliance[i - 1])]['on_power_threshold']:
            count_values[i - 1].append(j)

# 计算总的数据
mean = []
for i in range(len(appliance)):
    mae_mean = len(count_values[i])
    print('appliance:', appliance[i], 'sum:', mae_mean)
    mean.append(mae_mean)
    # print('appliance:', appliance[i], 'cnt:', count_values[i])

# 重新构造数据 取中位数
average = np.median(mean)
print(average)

indexed_mae_mean = list(enumerate(mean))
indexed_mae_mean.sort(key=lambda x: x[1])
min_indices = [index for index, _ in indexed_mae_mean[:3]]

for i in range(len(mean)):
    # print(test_set_copy[count_values[i]])
    mean_1 = np.mean(test_set_copy[count_values[i], :, i + 1])
    std = np.std(test_set_copy[count_values[i], :, i + 1])
    print('appliance:', appliance[i], 'mean:', mean_1, 'std:', std)

mean_1 = np.mean(test_set_copy[:, :, 0])
std = np.std(test_set_copy[:, :, 0])
print('aggregate:', 'mean:', mean_1, 'std:', std)


