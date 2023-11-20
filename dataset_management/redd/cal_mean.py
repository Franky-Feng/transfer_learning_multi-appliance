import numpy as np
import matplotlib.pyplot as plt
import Arguments

appliance = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
para = params_appliance = Arguments.redd_params_appliance
save_path = 'total/'

file_path = 'total/train_set.npy'
test_set = np.load(file_path)
test_set_copy = test_set.copy()

print(len(test_set_copy))
for j in range(len(test_set_copy)):
    test_set_copy[j, :, 0] = test_set_copy[j, :, 0] * params_appliance['aggregate']['std'] + params_appliance['aggregate']['mean']
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

for i in range(1, len(mean)):
    # print(test_set_copy[count_values[i]])
    mean_1 = np.mean(test_set_copy[count_values[i], :, i + 1])
    std = np.std(test_set_copy[count_values[i], :, i + 1])
    print('appliance:', appliance[i], 'mean:', mean_1, 'std:', std)

mean_1 = np.mean(test_set_copy[:, :, 0])
std = np.std(test_set_copy[:, :, 0])
print('aggregate:', 'mean:', mean_1, 'std:', std)

# 计算mean,std
# multi = []

# # 增加多少倍 数据少于多少，大于2数据太少
# for i in range(1,len(mean)):
#     multi.append(int(average / mean[i]))
#
# print(multi)
# # 减少多少倍 数据多于多少，大于2数据太多
# div = []
# for i in range(1,len(mean)):
#     div.append(int(mean[i] / average))
#
# print(div)
# # 处理数据太少的
# for a in range(len(multi)):
#     if multi[a] > 2:
#         # 复制索引值5次并添加到 test_set 后面
#         for i in count_values[a]:
#             repeated_indices = np.tile(test_set[i], (multi[a] - 2, 1, 1))
#             test_set = np.concatenate([test_set, repeated_indices], axis=0)
#
# # 处理数据太多的
# for a in range(len(div)):
#     if div[a] > 2:
#         new = np.setdiff1d(count_values[a], count_values[min_indices[0]])
#         new = np.setdiff1d(new, count_values[min_indices[1]])
#         new = np.setdiff1d(new, count_values[min_indices[2]])
#         random_indices = np.random.choice(new, size=int((div[a]-2)*average), replace=False)
#         test_set_after_removal = np.delete(test_set, random_indices, axis=0)
#
# #重新打乱
# num_datasets = test_set_after_removal.shape[0]
# indices = np.arange(num_datasets)
# np.random.shuffle(indices)
# shuffled_data_3d = test_set_after_removal[indices]
#
# # 保存数据
# np.save(save_path + 'train_set2.npy', shuffled_data_3d)
# print(len(test_set_after_removal))
