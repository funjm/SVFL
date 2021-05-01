import os
import random
import shutil
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#--------------将原数据集(2000)分为训练集(1600)测试集(400),存在split文件夹下

# 批处理
file_list = ['mfeat-fou', 'mfeat-fac', 'mfeat-kar', 'mfeat-pix', 'mfeat-zer', 'mfeat-mor']

# 原数据地址
# file_list = ['mfeat-fou']
# file_list = ['mfeat-fac']
# file_list = ['mfeat-kar']
# file_list = ['mfeat-mor']

DATASIZE = 2000
# !!!如果改变此参数,datasets中的也需要改
TRAIN_PRECENT = 0.8


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def split_datasets(file_name):
    dataset_file = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "Handwritten", 'mfeat', file_name))
    if not os.path.exists(dataset_file):
        raise Exception("\n{} 不存在，请下载 mfeat 放到\n{} 下，并解压即可".format(
            dataset_file, os.path.dirname(dataset_file)))
    print(dataset_file)

    # 分割后存放的文件夹
    split_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "Handwritten", 'mfeat', "splite", file_name))
    print(split_dir)
    makedir(split_dir)

    train_data_path = os.path.join(split_dir, "train_data.npy")
    test_data_path = os.path.join(split_dir, "test_data.npy")
    train_data = list()
    test_data = list()
    train_size_per_class = DATASIZE * TRAIN_PRECENT / 10

    df = pd.read_csv(dataset_file, header=None, sep='\s+')

    # # z - score归一化
    df = (df - df.mean()) / (df.std())

    npdata = np.array(df, dtype=np.float32)
    for i, data in enumerate(npdata):
        if (i % 200) < train_size_per_class:
            train_data.append(data)
        else:
            test_data.append(data)

    np.save(train_data_path, train_data)
    np.save(test_data_path, test_data)

    train_data = np.load(train_data_path)
    print('train_data\n', train_data)
    print(type(train_data), len(train_data))

    test_data = np.load(test_data_path)
    print("test_data:\n", test_data)
    print(type(test_data), len(test_data))

if __name__ == '__main__':
    for i in file_list:
        split_datasets(i)