import numpy as np
import torch
import os
import sys
import random
from PIL import Image
from torch.utils.data import Dataset


DATASIZE = 2000
# !!!如果改变此参数,split_datasets中的也需要改
TRAIN_PRECENT = 0.8
CLASS_NUM = 10


class handWritten_Dataset(Dataset):
    def __init__(self, data_path, transform=None, train=True):
        """
        handWritten的Dataset
        :param data_path: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.class_size = self.get_class_size(train)
        self.data_info = self.get_img_info(data_path)
        self.transform = transform


    def __getitem__(self, index):
        data, label = self.data_info[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, label, index

    def __len__(self):
        return len(self.data_info)

    # @staticmethod
    def get_class_size(self, train):
        class_size = int(DATASIZE*TRAIN_PRECENT/CLASS_NUM)
        if not train:
            class_size = 200 - class_size
        return class_size

    # @staticmethod
    def get_img_info(self, file_path):
        data_info = list()
        train_data = np.load(file_path)
        for i, data in enumerate(train_data):
            label = int(i / self.class_size)
            data_info.append((data, label))

        # print(data_info[0])
        return data_info

    def get_features_size(self):
        return len(self.data_info[0][0])

class test_Dataset(Dataset):
    def __init__(self, data_path, transform=None, train=True):
        """
        用于训练的Dataset
        :param data_path: list(str), 不同数据集所在路径的list
        :param transform: torch.transform，数据预处理
        :returns data_list: list(tensor), 每个tensor包含了([batchsize, feature_length])
                            同一个tensor中feature_length相同,不同tensor中feature_length不一定相同
        :returns label: tensor, 包含了batchsize个label.前class_size为0类,以此类推
        :returns index

        """
        self.data_path = data_path
        self.class_num = len(data_path)
        self.class_size = self.get_class_size(train)
        self.data_info = self.get_img_info(data_path)
        self.transform = transform

    def __getitem__(self, index):
        # print("index in __getitem__ is: ", index)
        data_list = list()
        label = None

        for i in range(self.class_num):
            data, label = self.data_info[i][index]
            # print(self.data_path[i])
            # print(data)
            data_list.append(data)

        return data_list, label, index

    def __len__(self):
        return len(self.data_info[0])

    # @staticmethod
    def get_class_size(self, train):
        #         (1-TRAIN_PRECENT) = 0.19999999999999996,所以不要用(1-TRAIN_PRECENT)
        class_size = int((DATASIZE - DATASIZE * TRAIN_PRECENT) / CLASS_NUM)
        print((TRAIN_PRECENT))
        print("class_size is:", class_size)
        return class_size

    # @staticmethod
    def get_img_info(self, file_path):
        data_info = list()
        for id, feature_set in enumerate(file_path):
            feature_info = list()
            train_data = np.load(feature_set)
            for i, data in enumerate(train_data):
                label = int(i / self.class_size)
                feature_info.append((data, label))
            data_info.append(feature_info)
            print("here is the {} feature set, the data[0] is: \n{}".format(id, data_info[id][0]))
        return data_info

    def get_features_size(self):
        featureset_length = list()
        for featureset in self.data_info:
            # 取featureset的第0个元素的数据.shape的第一个维度
            featureset_length.append(featureset[0][0].shape[0])
        return featureset_length

class common_representation_Dataset(Dataset):
    def __init__(self, dataset, transform=None, train=True):
        """
        mfeat-fou的Dataset
        :param dataset: str, 数据集
        :param transform: torch.transform，数据预处理
        """
        self.class_size = self.get_class_size(train)
        self.data_info = self.get_img_info(dataset)
        self.transform = transform

    def __getitem__(self, index):
        data, label = self.data_info[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, label, index

    def __len__(self):
        return len(self.data_info)

    # @staticmethod
    def get_class_size(self, train):
        class_size = int(DATASIZE*TRAIN_PRECENT/CLASS_NUM)
        if not train:
            class_size = 200 - class_size
        return class_size

    # @staticmethod
    def get_img_info(self, dataset):
        data_info = list()
        for i, data in enumerate(dataset):
            label = int(i / self.class_size)
            data_info.append((data, label))

        # print(data_info[0])
        return data_info

    def get_features_size(self):
        return len(self.data_info[0][0])


