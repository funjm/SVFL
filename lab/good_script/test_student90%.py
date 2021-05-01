import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
import model.MLP
from tools.datasets import test_Dataset
from tools.datasets import handWritten_Dataset
from tools.datasets import common_representation_Dataset

# !!!手动写
TRAIN_COMMON_REPRESENTATION_NUM = 1600

beta1 = 0.9
TEST_LR = 0.001
# 总类别个数
CLASS_NUM = 10
HIDDEN_LAYER = 1
TEST_BATCH_SIZE = 40
COMMON_REPRESENTATION_SIZE = 64
TRANSFERBRIDGE_BATCH_SIZE = 40
TEST_TRAINING_EPOCH = 1000

model_epoch = 19

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# p=2表示二范式, dim=1表示按行归一化,当tensor为1维时,dim=0
def tensor_normalize(tensor):
    if len(tensor.shape) == 1:
        norm = F.normalize(tensor, p=1, dim=0)
    else:
        norm = F.normalize(tensor, p=1, dim=1)
    return norm

# 批量创建teacher_model, TransferBridge, dataloader
# dataset_set = list()
# dataloader_set = list()
# test_dataset_set = list()
# test_dataloader_set = list()
# hidden_layer_set = list()
# teacher_model_set = list()
# transferBridge_set = list()


print("# ============================ step 1 load model & data ============================")
featureSet_list = ['mfeat-fou', 'mfeat-fac', 'mfeat-kar', 'mfeat-pix', 'mfeat-zer']

# 创建testdataset
testdataset_list = ['../datasets/Handwritten/mfeat/hidden_layers/test-'
                    + file_name + str(HIDDEN_LAYER) + '.npy' for file_name in featureSet_list]
testdataset = test_Dataset(testdataset_list)
# 获取dataset相关信息
com_representation_size_list = testdataset.get_features_size()
print("com_representation_size_list is: ", com_representation_size_list)

training_samples_size = TRAIN_COMMON_REPRESENTATION_NUM
featureSet_size = int(training_samples_size/CLASS_NUM)
other_featureSet_size = training_samples_size - featureSet_size
weight = (featureSet_size + other_featureSet_size) / featureSet_size / other_featureSet_size


test_dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# 1.创建student网络,并初始化
student_net = model.MLP.StudentMLP(classes=10, com_representation_size=COMMON_REPRESENTATION_SIZE).to(device).eval()
model_saved_path = "../saved_model/student_net" + str(model_epoch) + ".pkl"
state_dict_load = torch.load(model_saved_path)
student_net.load_state_dict(state_dict_load)
print("加载后student的参数: ", student_net.fc1.weight[0, ...])



# load transferBridge model
transferBridge_criterion = nn.MSELoss()

transferBridge_set = list()
for i in range(len(featureSet_list)):
    intermediate_size = com_representation_size_list[i]
    transferBridge_set.append(
        model.MLP.TransferBridge(intermediate_size, com_representation_size=COMMON_REPRESENTATION_SIZE).to(device).eval())
    model_saved_path = "../saved_model/transferBridge_to_" \
                       + featureSet_list[i] + str(model_epoch) + ".pkl"
    state_dict_load = torch.load(model_saved_path)
    transferBridge_set[i].load_state_dict(state_dict_load)

# load common_representation
model_saved_path = "../saved_model/np_common_representation" + str(model_epoch) + ".npy"
np_common_representation = np.load(model_saved_path)
common_representation = torch.tensor(np_common_representation, dtype=torch.float32).to(device)

comment = "TEST" + str(COMMON_REPRESENTATION_SIZE)
filename_name = comment
writer = SummaryWriter(comment=comment, filename_suffix=filename_name, log_dir="./test_runs")
print("# ============================ step 2 test  ============================")
correct = 0.
student_correct = 0.
common_representation_correct = 0.
total = 0.
print_count = 1
iter_count = 0

predict_list = list()

# TODO(先生成序号,再通过序号获得,方便处理样本不完全的情况)

for batch_idx, data in enumerate(test_dataloader):
    hidden_layers, labels, indexes = data

    if print_count:
        print_count = 0
        print("hidden_layers is \n", hidden_layers)
        for i in range(len(featureSet_list)):
            print(hidden_layers[i].shape)
        print("labels is \n", labels)

    hidden_layers, labels, indexes = [sample.to(device) for sample in hidden_layers], labels.to(device), indexes.to(
        device)

    # 随机生成共同表示
    test_common_representation = torch.rand((TEST_BATCH_SIZE, COMMON_REPRESENTATION_SIZE)
                                            , device=device, requires_grad=True)
    test_optimizer = optim.Adam([test_common_representation], lr=TEST_LR, betas=(beta1, 0.999))
    # test_optimizer = optim.SGD([test_common_representation], lr=TEST_LR, momentum=0.9)
    test_scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer, step_size=600, gamma=0.5)

    print("========== phase 4.2: training ==========")
    # train共同表示直到test_loss收敛:
    for converge in range(TEST_TRAINING_EPOCH):
        # 用featureSet_loss记录每个featureSet的loss,最后加起来backward
        featureSet_loss = torch.zeros((len(featureSet_list)), dtype=torch.float32).to(device)
        for i in range(len(featureSet_list)):
            outputs = transferBridge_set[i](test_common_representation)
            featureSet_loss[i] = transferBridge_criterion(hidden_layers[i], outputs)

        test_loss = torch.sum(featureSet_loss)
        # print("test_common_representation before back")
        test_loss.backward()
        test_optimizer.step()
        test_scheduler.step()

        # record&print
        if converge % 20 == 0:
            print("test loss: ", test_loss)
        if converge % 100 == 0:
            print("============================")
        # TEST_TRAINING_EPOCH+10方便观察
        iter_count += 1
        # 由于是从第101个epoch开始测试所以tensorboard上从404k开始记录
        writer.add_scalar("Test Loss", test_loss, iter_count)

    # prediction = student_net(共同表示) - 共同表示与各个类的contrastive_loss
    # !!!注意共同表示与各个类的contrastive_loss为负,所以需要减
    print("========== phase 4.2: predict ==========")
    pre_socre = student_net(test_common_representation)
    pre_socre = tensor_normalize(pre_socre)
    total_score = tensor_normalize(pre_socre)
    print("student predict score: \n", pre_socre[0])

    for i, sample in enumerate(test_common_representation):
        contrastive_loss_set = torch.zeros(CLASS_NUM, dtype=torch.float32).to(device)
        for label_i in range(CLASS_NUM):
            # print(featureSet_size)
            label_list = [i + featureSet_size * label_i for i in range(featureSet_size)]
            all_samples = torch.sum(sample * common_representation) / other_featureSet_size
            class_samples = torch.sum(sample * common_representation[label_list]) * weight
            contrastive_loss = all_samples - class_samples
            contrastive_loss_set[label_i] = contrastive_loss
            # 将contrastive_loss加到对应类别的分数上
        norm_contrastive_loss_set = tensor_normalize(contrastive_loss_set)
        total_score[i] -= norm_contrastive_loss_set
        # print("the contrastive ", norm_contrastive_loss_set)

    _, predicted = torch.max(total_score.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).cpu().squeeze().sum().numpy()
    print("total_predicted: ", (predicted == labels).sum())
    print(predicted == labels)

    _, student_predicted = torch.max(pre_socre.data, 1)
    student_correct += (student_predicted == labels).cpu().squeeze().sum().numpy()
    print("student_predicted: ", (student_predicted == labels).sum())
    print(student_predicted == labels)

    com_predicted = (pre_socre-total_score)
    _, common_representation_predicted = torch.max(com_predicted.data, 1)
    common_representation_correct += (common_representation_predicted == labels).cpu().squeeze().sum().numpy()
    print("common_representation_predicted: ", (common_representation_predicted == labels).sum())
    print(common_representation_predicted == labels)

    print(" This turn test_Acc:{:.2%}".format((predicted == labels).cpu().squeeze().sum().numpy()/labels.size(0)))
    writer.add_scalar("Accuracy_Test", correct / total, iter_count)
    print(" Loss: {:.4f} test_Acc:{:.2%}".format(test_loss, correct / total))
