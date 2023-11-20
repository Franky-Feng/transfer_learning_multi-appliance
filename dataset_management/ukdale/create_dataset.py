import pandas as pd
import time
import os
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import Arguments

DATA_DIRECTORY = '../../data/uk_dale/'
SAVE_PATH = 'total/'

para = params_appliance = Arguments.ukdale_params_appliance
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


house_indicies = [1, 2, 5]

appliance_name = [['kettle', 'microwave', 'fridge', 'dishwasher', 'washing_machine'],
                  ['kettle', 'microwave', 'fridge', 'dish_washer', 'washing_machine'],
                  ['kettle', 'microwave', 'fridge_freezer', 'dishwasher', 'washer_dryer']]

appliance_name2 = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washing_machine']


def main():
    args = get_arguments()

    path = args.data_dir
    save_path = args.save_path

    print(path)
    aggregate_mean = args.aggregate_mean
    aggregate_std = args.aggregate_std

    print("Starting creating dataset...")
    # Looking for proper files
    cnt = 0
    for house_id in house_indicies:
        cnt += 1
        house_folder = path + 'house_' + str(house_id)
        house_label = pd.read_csv(house_folder + '\labels.dat', sep=' ', header=None)

        house_data = pd.read_csv(house_folder + '\channel_1.dat', sep=' ', header=None)
        house_data.iloc[:, 0] = pd.to_datetime(house_data.iloc[:, 0], unit='s')
        house_data.columns = ['time', 'aggregate']
        house_data = house_data.set_index('time')
        house_data = house_data.resample('6s').mean().fillna(method='ffill', limit=30)

        appliance_list = house_label.iloc[:, 1].values
        app_index_dict = defaultdict(list)

        for appliance in appliance_name[cnt-1]:
            data_found = False
            for i in range(len(appliance_list)):
                if appliance_list[i] == appliance:
                    app_index_dict[appliance].append(i + 1)
                    data_found = True

            if not data_found:
                app_index_dict[appliance].append(-1)

            # print(np.sum(list(app_index_dict.values())))
            # if np.sum(list(app_index_dict.values())) == -len(appliance_name):
            #     house_indicies.remove(house_id)
            #     continue
        flag = 0
        for appliance in appliance_name[cnt-1]:

            if app_index_dict[appliance][0] == -1:
                house_data.insert(len(house_data.columns), appliance, np.zeros(len(house_data)))
            else:
                temp_data = pd.read_csv(house_folder + '\channel_' + str(app_index_dict[appliance][0]) + '.dat',
                                        sep=' ', header=None)
                temp_data.iloc[:, 0] = pd.to_datetime(temp_data.iloc[:, 0], unit='s')
                temp_data.columns = ['time', appliance_name2[flag]]
                temp_data = temp_data.set_index('time')
                temp_data = temp_data.resample('6s').mean().fillna(method='ffill', limit=30)
                house_data = pd.merge(house_data, temp_data, how='inner', on='time')
                flag +=1
        if house_id == house_indicies[0]:
            entire_data = house_data
            if len(house_indicies) == 1:
                entire_data = entire_data.reset_index(drop=True)
        else:
            entire_data = pd.concat([entire_data, house_data], ignore_index=True)
            # entire_data = entire_data.append(house_data, ignore_index=True)
    entire_data = entire_data.dropna().copy()
    entire_data = entire_data[entire_data['aggregate'] > 0]
    entire_data[entire_data < 5] = 0
    # total_csv = entire_data.clip([0] * len(entire_data.columns), cutoff, axis=1)
    total_csv = entire_data
    total_csv['aggregate'] = (total_csv['aggregate'] - aggregate_mean) / aggregate_std
    for i in appliance_name2:
        total_csv[i] = (total_csv[i] - params_appliance[i]['mean']) / params_appliance[i]['std']

    chunk_size = 240
    total_data = total_csv.values  # 将 DataFrame 转换为 NumPy 数组
    num_rows = len(total_data)
    data_sets = [total_data[i:i + 480, :] for i in range(0, num_rows - 480, chunk_size)]

    # 将数据集组合成一个三维数组
    x = len(data_sets)
    data_3d = np.stack(data_sets)
    # shuffle
    num_datasets = data_3d.shape[0]
    indices = np.arange(num_datasets)
    np.random.shuffle(indices)
    shuffled_data_3d = data_3d[indices]

    # 计算划分数据集的比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # 计算划分的索引
    train_end = int(x * train_ratio)
    val_end = train_end + int(x * val_ratio)

    # 划分数据集
    train_set = shuffled_data_3d[:train_end]
    val_set = shuffled_data_3d[train_end:val_end]
    test_set = shuffled_data_3d[val_end:]

    # save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + 'train_set.npy', train_set)
    np.save(save_path + 'val_set.npy', val_set)
    np.save(save_path + 'test_set.npy', test_set)


if __name__ == '__main__':
    main()
