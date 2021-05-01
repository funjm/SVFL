import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
import model.MLP
from tools.datasets import handWritten_Dataset
from tools.datasets import common_representation_Dataset
# data parameters setting
CLASS = 10
COMMON_REPRESENTATION_SIZE = 100
# 逼近的hidden layer的层数
HIDDEN_LAYER = 2
# training parameters setting
MAX_EPOCH = 8000
BATCH_SIZE = 40
LR = 0.0002
# loss记录间隔
log_interval = 10
val_interval = 1
beta1 = 0.5
DATASIZE = 2000
CLASS_NUM = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("# ============================ step 1/6 student_model, teacher_model, hidden_layer, TransferBridge, dataloader ============================")
# 1.创建student网络,并初始化
student_net = model.MLP.StudentMLP(classes=10, com_representation_size=COMMON_REPRESENTATION_SIZE)
student_net.to(device)
student_net.initialize_weights()

# 2.批量创建teacher_model, TransferBridge, dataloader
featureSet_list = ['mfeat-fou', 'mfeat-fac', 'mfeat-kar', 'mfeat-pix', 'mfeat-zer']

dataset_set = list()
dataloader_set = list()
hidden_layer_set = list()
teacher_model_set = list()
transferBridge_set = list()
for i, file_name in enumerate(featureSet_list):
    train_file_path = '../datasets/Handwritten/mfeat/splite/' + file_name + '/train_data.npy'
    model_path = "../saved_model/" + file_name + ".pkl"
    hidden_layers_path = '../datasets/Handwritten/mfeat/hidden_layers/' + file_name + str(HIDDEN_LAYER) + '.npy'

    # hidden_layer
    hidden_layer_set.append(torch.tensor(np.load(hidden_layers_path)).to(device))
    print("hidden layer's shape: ", hidden_layer_set[i].shape)

    # dataset&dataloader
    dataset_set.append(handWritten_Dataset(data_path=train_file_path))
    dataloader_set.append(torch.utils.data.DataLoader(dataset=dataset_set[i], batch_size=BATCH_SIZE, shuffle=True))

    # load teacher_model_set
    teacher_model_set.append(model.MLP.TeacherMLP(classes=10, input_size=dataset_set[i].get_features_size()).to(device))
    state_dict_load = torch.load(model_path)
    teacher_model_set[i].load_state_dict(state_dict_load)
    print("加载后teacher的参数: ", teacher_model_set[i].fc1.weight[0, ...])

    # TransferBridge,并初始化
    intermediate_size = hidden_layer_set[i].shape[1]
    transferBridge_set.append(model.MLP.TransferBridge(intermediate_size, com_representation_size=COMMON_REPRESENTATION_SIZE).to(device))
    print("\ntransferBridge: ", transferBridge_set[i])
    transferBridge_set[i].initialize_weights()


print("# ============================ step 2/6 generate common representation & it's dataloader ============================")
training_samples_size = (len(dataset_set[0]))
common_representation = torch.rand((training_samples_size, COMMON_REPRESENTATION_SIZE)
                                   , device=device, requires_grad=True)
print(common_representation)
print(common_representation.size())

common_representation_dataset = common_representation_Dataset(dataset=common_representation)
common_representation_dataloader = torch.utils.data.DataLoader(dataset=common_representation_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("# ============================ step 3/6 loss function ============================")
student_net_criterion = nn.CrossEntropyLoss()
transferBridge_criterion = nn.MSELoss()

print("# ============================ step 4/6 optimizer ============================")
# optimizer for student
student_net_optimizer = optim.Adam(student_net.parameters(), lr=LR,  betas=(beta1, 0.999))
student_net_scheduler = torch.optim.lr_scheduler.StepLR(student_net_optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

# optimizer for transferBridge
transferBridge_optimizer_set = list()
transferBridge_scheduler_set = list()
for i in range(len(featureSet_list)):
    transferBridge_optimizer_set.append(optim.Adam(transferBridge_set[i].parameters(), lr=0.00001*LR, betas=(beta1, 0.999)))
    transferBridge_scheduler_set.append(torch.optim.lr_scheduler.StepLR(transferBridge_optimizer_set[i], step_size=10, gamma=0.1))  # 设置学习率下降策略

# optimizer for common representation
common_representation_optimizer = optim.Adam([common_representation], lr=LR, betas=(beta1, 0.999))
common_representation_scheduler = torch.optim.lr_scheduler.StepLR(common_representation_optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

print("# ============================ step 5/6 recording ============================")
# 记录总迭代次数,方便tensorboard记录
transferBridge_iter_count = [0, 0, 0, 0, 0]
student_iter_count = 0
common_representation_iter_count = 0

comment = " STUDENT"+"_MAX-EPOCH:"+str(MAX_EPOCH)+"_LR:"+str(LR) + "_COM-REP-SIZE:" + str(COMMON_REPRESENTATION_SIZE)
filename_name = comment
writer = SummaryWriter(comment=comment, filename_suffix=filename_name)

print("# ============================ step 6/6 training ============================")
train_curve = list()

for epoch in range(MAX_EPOCH):
    print(" ============================ the {}th epoch  ============================".format(epoch))
    loss_mean = 0.
    correct = 0.
    total = 0.

    print("================== phase 1: training transfer bridge ==================")
    for i in range(len(featureSet_list)):
        # print("----------- calculate the {} transfer bridge -----------".format(i))
        for batch_idx, data in enumerate(dataloader_set[i]):
            transferBridge_iter_count[i] += 1
            # 注意这里取出来的就是indexes,labels即hidden layer直接通过序号去读取好的hidden layer set中获取
            # outputs需要通过indexes获取共同表示中,并通过transferBridge计算得到
            _, _, indexes = data
            labels = hidden_layer_set[i][indexes]
            outputs = transferBridge_set[i](common_representation[indexes])
            # print("-----------------------hidden layer is:\n", labels.shape, labels)
            # print("-----------------------outputs is:\n", outputs.shape, outputs)

            transferBridge_optimizer_set[i].zero_grad()
            loss = transferBridge_criterion(outputs, labels)
            loss.backward()
            transferBridge_optimizer_set[i].step()

            writer.add_scalar("Train {} transfer_bridge".format(featureSet_list[i]), loss.item(), transferBridge_iter_count[i])


        # transferBridge_scheduler_set[i].step()

    print("================== phase 2: training student model ==================")
    for batch_idx, data in enumerate(common_representation_dataloader):
        student_iter_count += 1
    # # 用第三个训练集kar(同64特征)也可以训练
    # for batch_idx, data in enumerate(dataloader_set[2]):

        inputs, labels, indexes = data
        inputs, labels, indexes = inputs.to(device), labels.to(device), indexes.to(device)

        outputs = student_net(inputs)

        student_net_optimizer.zero_grad()
        loss = student_net_criterion(outputs, labels)
        loss.backward()
        student_net_optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).cpu().squeeze().sum().numpy()
        # 记录
        writer.add_scalar("Train student_model_Acc:", correct / total, student_iter_count)

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (batch_idx + 1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] \Loss_per_record_interval: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, batch_idx + 1, len(common_representation_dataloader), loss_mean, correct / total))
            loss_mean = 0.

        writer.add_scalar("Train student_model_loss", loss.item(), student_iter_count)

    # student_net_scheduler.step()

    print("================== phase 3: training common representation ==================")
    cal_contrastive_loss_start = time.time()
    contrastive_loss = torch.tensor(0, dtype=torch.float32).to(device)
    # 每个类别样本的数量
    featureSet_size = int(training_samples_size/CLASS_NUM)

    common_representation_iter_count += 1
    # for i, sample in enumerate(common_representation):
    #     label_i = int(i / featureSet_size)
    #
    #     for j, other_sample in enumerate(common_representation):
    #         if i == j:
    #             continue
    #         label_j = int(j / featureSet_size)
    #
    #         # 最小化loss, 异类loss值为正,同类loss值为负
    #         if label_i == label_j:
    #             # 对应位置相乘,再求和
    #             contrastive_loss -= torch.sum(sample * other_sample) / featureSet_size
    #         else:
    #             contrastive_loss += torch.sum(sample * other_sample) / (training_samples_size-featureSet_size)
    #
    #     print('\r calculate contrastive loss completion precent is {}%'.format(i/training_samples_size*100), end='')
    other_featureSet_size = training_samples_size - featureSet_size
    weight = (featureSet_size + other_featureSet_size) / featureSet_size / other_featureSet_size


    for i, sample in enumerate(common_representation):
        label_i = int(i / featureSet_size)
        # print(featureSet_size)
        label_list = [i + featureSet_size * label_i for i in range(featureSet_size)]
        all_samples = torch.sum(sample * common_representation) / other_featureSet_size
        class_samples = torch.sum(sample * common_representation[label_list]) * weight
        self_samples = torch.sum(sample * sample) / featureSet_size
        contrastive_loss += all_samples - class_samples + self_samples

    cal_contrastive_loss_end = time.time()
    print("\nit takes {}s to calculate contrastive loss".format(str(cal_contrastive_loss_end - cal_contrastive_loss_start)))

    contrastive_loss /= common_representation.size()[0]
    writer.add_scalar("Train contrastive_loss", contrastive_loss.item(), common_representation_iter_count)

    backward_start = time.time()
    contrastive_loss.backward()
    backward_end = time.time()
    print("\nit takes {}s to backward".format(str(backward_end - backward_start)))
    common_representation_optimizer.step()

    # 由于需要其他phase的grad,所以在这里清零梯度
    common_representation_optimizer.zero_grad()

    # common_representation_scheduler.step()

train_x = range(len(train_curve))
train_y = train_curve

plt.plot(train_x, train_y, label='Train')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()