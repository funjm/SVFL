import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
# from model.MLP import TeacherMLP
import model.MLP
from tools.datasets import handWritten_Dataset

BATCH_SIZE = 40

# 需要获取HIDDEN_LAYER的层数,list(net.modules())[HIDDEN_LAYER],list(net.modules())[0]是概览
HIDDEN_LAYER = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 批处理
featureSet_list = ['mfeat-fou', 'mfeat-fac', 'mfeat-kar', 'mfeat-pix', 'mfeat-zer']


teacher_model_set = list()
train_dataset_set = list()
train_dataloader_set = list()

test_dataset_set = list()
test_dataloader_set = list()
for i, file_name in enumerate(featureSet_list):
    train_file_path = '../datasets/Handwritten/mfeat/splite/' + file_name + '/train_data.npy'
    test_file_path = '../datasets/Handwritten/mfeat/splite/' + file_name + '/test_data.npy'
    model_path = "../saved_model/" + file_name + ".pkl"

    # dataset&dataloader
    train_dataset_set.append(handWritten_Dataset(data_path=train_file_path, train=False))
    test_dataset_set.append(handWritten_Dataset(data_path=test_file_path))
    # 只是获取hidden layer,而发训练,不需要打乱
    train_dataloader_set.append(torch.utils.data.DataLoader(dataset=train_dataset_set[i], batch_size=BATCH_SIZE, shuffle=False))
    test_dataloader_set.append(torch.utils.data.DataLoader(dataset=test_dataset_set[i], batch_size=BATCH_SIZE, shuffle=False))

    # load teacher_model_set
    teacher_model_set.append(model.MLP.TeacherMLP(classes=10, input_size=train_dataset_set[i].get_features_size()).to(device))
    state_dict_load = torch.load(model_path)
    teacher_model_set[i].load_state_dict(state_dict_load)
    print("加载后: ", teacher_model_set[i].fc1.weight[0, ...])

print("================== get hidden layer ==================")
with torch.no_grad():
    print(len(featureSet_list))
    for i in range(len(featureSet_list)):
        net = teacher_model_set[i]
        train_loader = train_dataloader_set[i]

        # ==================使用hook记录中间层输出
        train_hidden_block = list()
        def train_forward_hook(module, data_input, data_output):
            # print(data_output.shape)
            train_hidden_block.append(data_output)

        list(net.modules())[HIDDEN_LAYER].register_forward_hook(train_forward_hook)

        # 主要是为了过一遍训练集,让hook函数记录
        for batch_idx, data in enumerate(train_loader):
            inputs, _, _ = data
            inputs = inputs.to(device)
            outputs, _, _ = net(inputs)

        # 从train_hidden_block中提取hidden,保存
        ndarray = list()
        for sample in train_hidden_block:
            ndarray.append(sample.cpu().numpy())
        ndarray = np.array(ndarray)
        ndarray = ndarray.reshape(-1, train_hidden_block[0].shape[-1])
        hidden_layers_path = '../datasets/Handwritten/mfeat/hidden_layers/' \
                             + featureSet_list[i] + str(HIDDEN_LAYER) + '.npy'
        print(ndarray.shape)
        print(hidden_layers_path)
        np.save(hidden_layers_path, ndarray)


with torch.no_grad():
    print(len(featureSet_list))
    for i in range(len(featureSet_list)):
        net = teacher_model_set[i]
        test_loader = test_dataloader_set[i]

        # ==================使用hook记录中间层输出
        test_hidden_block = list()
        def test_forward_hook(module, data_input, data_output):
            # print(data_output.shape)
            test_hidden_block.append(data_output)

        list(net.modules())[HIDDEN_LAYER].register_forward_hook(test_forward_hook)

        # 主要是为了过一遍test集,让hook函数记录
        for batch_idx, data in enumerate(test_loader):
            inputs, _, _ = data
            inputs = inputs.to(device)
            outputs, _, _ = net(inputs)

        # 从test_hidden_block中提取hidden,保存
        ndarray = list()
        for sample in test_hidden_block:
            ndarray.append(sample.cpu().numpy())
        ndarray = np.array(ndarray)
        print("test hidden layer's shape before reshape:", ndarray.shape)
        ndarray = ndarray.reshape(-1, test_hidden_block[0].shape[-1])
        print("test hidden layer's shape after reshape:", ndarray.shape)
        hidden_layers_path = '../datasets/Handwritten/mfeat/hidden_layers/' \
                            +"test-" + featureSet_list[i] + str(HIDDEN_LAYER) + '.npy'
        print(hidden_layers_path)
        np.save(hidden_layers_path, ndarray)
