import pandas as pd
import time
import os
import re
import argparse
import numpy as np

import Arguments

DATA_DIRECTORY = '../../data/refit/CLEAN_REFIT_081116/'
SAVE_PATH = 'total/'

para = params_appliance = Arguments.refit_params_appliance
AGG_MEAN = params_appliance['aggregate']['mean']
AGG_STD = params_appliance['aggregate']['std']


def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to sequence learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the CLEAN REFIT data')
    parser.add_argument('--aggregate_mean', type=int, default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std', type=int, default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                        help='The directory to store the training data')
    return parser.parse_args()


houses_list = [2, 3, 5, 9, 11, 20]
# 每间房里应该读哪些channel ，按照kettle，microwave，fridge，dishwasher，washingmachine的顺序
# -1表示没有
appliance_list = {
    '2': [8, 5, 1, 3, 2],
    '3': [9, 8, 2, 5, 6],
    '5': [8, 7, 1, 4, 3],
    '9': [7, 6, 1, 4, 3],
    '11': [7, 6, 1, 4, 3],
    '20': [9, 8, 1, 5, 4]

}
appliance_name = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']


def load(path, building, appliance_list):
    # load csv
    file_name = path + 'CLEAN_House' + str(building) + '.csv'
    # 根据appliance提供相关读取数据
    name = ['aggregate']
    channel = [2]
    zero_channel = []
    for i in range(len(appliance_list[str(building)])):
        c = appliance_list[str(building)][i]
        if c != -1:
            name.append(appliance_name[i])
            channel.append(2 + c)
        else:
            zero_channel.append(i)
    single_csv = pd.read_csv(file_name, header=0,
                             names=name,
                             usecols=channel,
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True
                             )
    # 补0
    for i in zero_channel:
        single_csv.insert(i + 1, appliance_name[i], 0)
    return single_csv


def main():
    start_time = time.time()
    # test path
    # path = '../../../data/refit/CLEAN_REFIT_081116/'
    # save_path = 'refitdata/'

    args = get_arguments()

    path = args.data_dir
    save_path = args.save_path

    print(path)
    aggregate_mean = args.aggregate_mean
    aggregate_std = args.aggregate_std

    total_length = 0
    print("Starting creating dataset...")
    # Looking for proper files
    for idx, filename in enumerate(os.listdir(path)):
        single_step_time = time.time()
        if int(re.search(r'\d+', filename).group()) in houses_list:
            print('File: ' + filename)
            csv = load(path, re.search(r'\d+', filename).group(), appliance_list)
            if 'total_csv' not in locals() and 'total_csv' not in globals():
                total_csv = csv
            else:
                total_csv = pd.concat([total_csv, csv], ignore_index=True)

    # # Normalization 没转之前
    # print('没转之前')
    # print(total_csv['aggregate'])
    # variance = total_csv['aggregate'].std()
    # mean_value = total_csv['aggregate'].mean()
    # print("aggregate_variance：", variance)
    # print("aggregate_mean_value：", mean_value)
    # variance = total_csv['kettle'].std()
    # mean_value = total_csv['kettle'].mean()
    # print("kettle_variance：", variance)
    # print("kettle_mean_value：", mean_value)

    total_csv['aggregate'] = (total_csv['aggregate'] - aggregate_mean) / aggregate_std
    for i in appliance_name:
        total_csv[i] = (total_csv[i] - params_appliance[i]['mean']) / params_appliance[i]['std']
    #
    # # 转回去
    # total_csv['aggregate'] = total_csv['aggregate'] * aggregate_std + aggregate_mean
    # for i in appliance_name:
    #     total_csv[i] = total_csv[i] * params_appliance[i]['std'] + params_appliance[i]['mean']

    # print('转了')
    # print(total_csv['aggregate'])
    # variance = total_csv['aggregate'].std()
    # mean_value = total_csv['aggregate'].mean()
    # print("aggregate方差：", variance)
    # print("aggregate平均值：", mean_value)
    # variance = total_csv['kettle'].std()
    # mean_value = total_csv['kettle'].mean()
    # print("kettle方差：", variance)
    # print("kettle平均值：", mean_value)

    chunk_size = 240
    total_data = total_csv.values  # 将 DataFrame 转换为 NumPy 数组
    num_rows = len(total_data)
    data_sets = [total_data[i:i + 480, :] for i in range(0, num_rows - 480, chunk_size)]

    # 将数据集组合成一个三维数组
    data_3d = np.stack(data_sets)
    # shuffle
    num_datasets = data_3d.shape[0]
    indices = np.arange(num_datasets)
    np.random.shuffle(indices)
    shuffled_data_3d = data_3d[indices]

    # 数据清洗
    new_shuffled_data_3d = []
    print('len:', len(shuffled_data_3d))
    for data in shuffled_data_3d:
        data[:, 0] = data[:, 0] * aggregate_std + aggregate_mean
        for i in range(1, len(appliance_name) + 1):
            data[:, i] = data[:, i] * params_appliance[appliance_name[i - 1]]['std'] + \
                         params_appliance[appliance_name[i - 1]]['mean']

        # 计算data[:, 1]到data[:, 5]的和
        sum_data_1_to_5 = np.sum(data[:, 1:6], axis=1)
        result = data[:, 0] - sum_data_1_to_5
        count_negative_values = np.sum(result < 0)
        # 如果小于0的值的数量不超过10，则将data添加到新列表中
        if np.sum(result < 0) <= 1:
            new_shuffled_data_3d.append(data)

    print('len:', len(new_shuffled_data_3d))
    new_shuffled_data_3d = np.stack(new_shuffled_data_3d)
    new_shuffled_data_3d[:, :, 0] = (new_shuffled_data_3d[:, :, 0] - aggregate_mean) / aggregate_std
    for i in range(1, len(appliance_name) + 1):
        new_shuffled_data_3d[:, :, i] = (new_shuffled_data_3d[:, :, i] - params_appliance[appliance_name[i - 1]][
            'mean']) / params_appliance[appliance_name[i - 1]]['std']

    # 计算划分数据集的比例
    x = len(new_shuffled_data_3d)

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # 计算划分的索引
    train_end = int(x * train_ratio)
    val_end = train_end + int(x * val_ratio)

    # 划分数据集
    train_set = new_shuffled_data_3d[:train_end]
    val_set = new_shuffled_data_3d[train_end:val_end]
    test_set = new_shuffled_data_3d[val_end:]

    # save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + 'train_set.npy', train_set)
    np.save(save_path + 'val_set.npy', val_set)
    np.save(save_path + 'test_set.npy', test_set)

    print("\nNormalization parameters: ")
    print("Mean and standard deviation values USED for AGGREGATE are:")
    print("    Mean = {:d}, STD = {:d}".format(aggregate_mean, aggregate_std))
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))


if __name__ == '__main__':
    main()
