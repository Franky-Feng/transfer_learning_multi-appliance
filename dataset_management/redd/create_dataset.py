import pandas as pd
import time
import os
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import Arguments

DATA_DIRECTORY = '../../data/redd/'
SAVE_PATH = 'total/'
params_appliance = Arguments.redd_params_appliance
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


cutoff=[3998,3998,3998,3998,3998]
house_indicies = [1, 2, 3]

appliance_name = ['kettle', 'microwave', 'fridge', 'dish washer', 'washer dryer']


def load(path, building, appliance_list):
    # load csv
    file_name = path + 'building_' + str(building) + '.csv'
    new_df  = pd.DataFrame()
    single_csv = pd.read_csv(file_name, header=0,na_filter=False,parse_dates=True, infer_datetime_format=True,
                             memory_map=True)

    # aggregate
    row_sums = single_csv.sum(axis=1)
    new_df['aggregate']=row_sums
    for column  in appliance_name:
        if column in  single_csv.columns:
            new_df[column] = single_csv[column]
        else:
            print(len(single_csv))
            zero_padding = np.zeros(len(single_csv))
            zero_padding_series = pd.Series(zero_padding)
            new_df[column] = zero_padding_series
            # zero_padding = zero_padding.reshape(-1, 1)
            # new_df[column]=zero_padding
    return new_df


def main():
    args = get_arguments()

    path = args.data_dir
    save_path = args.save_path

    print(path)
    aggregate_mean = args.aggregate_mean  # 522
    aggregate_std = args.aggregate_std  # 814

    total_length = 0
    print("Starting creating dataset...")
    # Looking for proper files

    for idx, filename in enumerate(os.listdir(path)):
        if int(re.search(r'\d+', filename).group()) in house_indicies:
            print('File: ' + filename)
            csv = load(path, re.search(r'\d+', filename).group(), appliance_name)
            if 'total_csv' not in locals() and 'total_csv' not in globals():
                total_csv = csv
            else:
                total_csv = pd.concat([total_csv, csv], ignore_index=True)

    # entire_data = entire_data.dropna().copy()
    # entire_data = entire_data[entire_data['aggregate'] > 0]
    # entire_data[entire_data < 5] = 0




    # total_csv = entire_data.clip([0] * len(entire_data.columns), cutoff, axis=1)

    total_csv['aggregate'] = (total_csv['aggregate'] - aggregate_mean) / aggregate_std
    for i in appliance_name:
        total_csv[i] = (total_csv[i] - params_appliance[i]['mean']) / params_appliance[i]['std']

    chunk_size = 60
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
