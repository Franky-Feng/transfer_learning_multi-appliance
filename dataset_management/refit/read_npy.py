import numpy as np
import matplotlib.pyplot as plt
import Arguments

channel = 3
file_path = 'total/val_set2.npy'
test_set = np.load(file_path)
appliance = Arguments.appliance_name
appliance2 = ['aggregate', 'kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
plt.rcParams['font.family'] = 'SimSun'  # 使用宋体字体
test_set_copy = test_set.copy()
para = params_appliance = Arguments.refit_params_appliance
aggregate_mean = params_appliance['aggregate']['mean']
aggregate_std = params_appliance['aggregate']['std']
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

# for i in range(100):
#     for j in range(6):
#         data_to_plot = test_set_copy[i, :, j]  # 假设你想绘制第一个数据集的第一列数据
#         plt.plot(data_to_plot, label=f' {appliance2[j]}')
#     plt.xlabel('time')
#     plt.ylabel('value')
#     plt.title('example')
#     plt.legend()  # 添加图例以显示数据集和列信息
#     plt.show()



for num in range(10):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for i in range(9):  # 一共有16个子图
        row, col = divmod(i, 3)  # 计算当前子图的行和列

        for j in range(2):
            if appliance2[j] == 'aggregate':
                data_to_plot = test_set_copy[count_values[channel][i+num*9], :, j]  # 假设你想绘制第一个数据集的第一列数据
                axes[row, col].plot(data_to_plot, label=f'{appliance2[j]}', linestyle='--', color='gray')
            else:
                data_to_plot = test_set_copy[count_values[channel][i+num*9], :, channel + 1]  # 假设你想绘制第一个数据集的第一列数据
                axes[row, col].plot(data_to_plot, label=f'{appliance2[channel+1]}')
        axes[row, col].set_xlabel('time step')
        axes[row, col].set_ylabel('power')
        axes[row, col].legend()

    plt.tight_layout()  # 使子图之间的间距合适
    plt.show()