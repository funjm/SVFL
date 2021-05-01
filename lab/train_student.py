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
# data parameters setting
# 总类别个数
CLASS_NUM = 10
# 数据集总大小
DATASIZE = 2000

# 共同表示的size
# COMMON_REPRESENTATION_SIZE = 100
COMMON_REPRESENTATION_SIZE = 64
# 逼近的hidden layer的层数
HIDDEN_LAYER = 1
# training parameters setting
MAX_EPOCH = 8000
TEST_TRAINING_EPOCH = 600
BATCH_SIZE = 40
# 为了transferBridge能平稳训练,将batchsize改为1600
TRANSFERBRIDGE_BATCH_SIZE = 40
# 目前是200,一次性测完,若改成其他值,test代码需要相应改变,至少writer要
TEST_BATCH_SIZE = 40
LR = 0.001
transforms_bridge_RL = 0.0004
# beta1 = 0.5
beta1 = 0.9
TEST_LR = 0.01
weight_decay = 0.01
trans_bridge_weight_decay = 0.001
# weight_decay = 0.000001
# loss记录间隔
log_interval = 10
# 模型保存间隔
save_interval = 10
# 是否测试
# test_flag = True
test_flag = True
save_flag = False
val_interval = 1
epoch_begin_test = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("paramters's setting: CLASS_NUM:{}, COMMON_REPRESENTATION_SIZE:{}, HIDDEN_LAYER:{}, MAX_EPOCH:{},"
      "TEST_TRAINING_EPOCH:{}, BATCH_SIZE:{}, TRANSFERBRIDGE_BATCH_SIZE:{}, TEST_BATCH_SIZE:{}, "
      "LR:{}, transforms_bridge_RL:{},beta1:{}, TEST_LR:{}, weight_decay:{}, trans_bridge_weight_decay:{}"
      .format(CLASS_NUM, COMMON_REPRESENTATION_SIZE, HIDDEN_LAYER, MAX_EPOCH, TEST_TRAINING_EPOCH
              , BATCH_SIZE, TRANSFERBRIDGE_BATCH_SIZE, TEST_BATCH_SIZE, LR, transforms_bridge_RL,
               beta1, TEST_LR, weight_decay, trans_bridge_weight_decay))
print("log_interval:{}, save_interval{}, test_flag:{}, val_interval:{}, epoch_begin_test:{}, save_flag:{}"
      .format(log_interval, save_interval, test_flag, val_interval, epoch_begin_test, save_flag))

# p=2表示二范式, dim=1表示按行归一化,当tensor为1维时,dim=0
def tensor_normalize(tensor):
    if len(tensor.shape) == 1:
        norm = F.normalize(tensor, p=1, dim=0)
    else:
        norm = F.normalize(tensor, p=1, dim=1)
    return norm



print("# ============================ step 1/6 student_model, teacher_model, hidden_layer, TransferBridge, dataloader ============================")
# 1.创建student网络,并初始化
student_net = model.MLP.StudentMLP(classes=10, com_representation_size=COMMON_REPRESENTATION_SIZE)
student_net.to(device)
student_net.initialize_weights()

# 2.批量创建teacher_model, TransferBridge, dataloader
featureSet_list = ['mfeat-fou', 'mfeat-fac', 'mfeat-kar', 'mfeat-pix', 'mfeat-zer']

dataset_set = list()
dataloader_set = list()
test_dataset_set = list()
test_dataloader_set = list()
hidden_layer_set = list()
teacher_model_set = list()
transferBridge_set = list()
for i, file_name in enumerate(featureSet_list):
    train_file_path = '../datasets/Handwritten/mfeat/splite/' + file_name + '/train_data.npy'
    model_path = "../saved_model/" + file_name + ".pkl"
    hidden_layers_path = '../datasets/Handwritten/mfeat/hidden_layers/' + file_name + str(HIDDEN_LAYER) + '.npy'
    test_file_path = '../datasets/Handwritten/mfeat/splite/' + file_name + '/test_data.npy'

    # hidden_layer
    hidden_layer_set.append(torch.tensor(np.load(hidden_layers_path)).to(device))
    print("hidden layer's shape: ", hidden_layer_set[i].shape)

    # dataset&dataloader
    dataset_set.append(handWritten_Dataset(data_path=train_file_path))
    dataloader_set.append(torch.utils.data.DataLoader
                          (dataset=dataset_set[i], batch_size=TRANSFERBRIDGE_BATCH_SIZE, shuffle=True))

    teacher_model_set.append(model.MLP.TeacherMLP(classes=10, input_size=dataset_set[i].get_features_size()).to(device).eval())
    state_dict_load = torch.load(model_path)
    teacher_model_set[i].load_state_dict(state_dict_load)
    print("加载后teacher的参数: ", teacher_model_set[i].fc1.weight[0, ...])

    # TransferBridge,并初始化
    intermediate_size = hidden_layer_set[i].shape[1]
    transferBridge_set.append(model.MLP.TransferBridge(intermediate_size, com_representation_size=COMMON_REPRESENTATION_SIZE).to(device))
    print("\ntransferBridge: ", transferBridge_set[i])
    transferBridge_set[i].initialize_weights()

# 3.创建testdataset
testdataset_list = ['../datasets/Handwritten/mfeat/hidden_layers/test-'
                    + file_name + str(HIDDEN_LAYER) + '.npy' for file_name in featureSet_list]
testdataset = test_Dataset(testdataset_list)
test_dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=TEST_BATCH_SIZE, shuffle=True)


print("# ============================ step 2/6 generate common representation & it's dataloader ============================")
training_samples_size = (len(dataset_set[0]))
print("training_samples_size is: ", training_samples_size)
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
student_net_optimizer = optim.Adam(student_net.parameters(), lr=LR
                                   , betas=(beta1, 0.999), weight_decay=weight_decay)
student_net_scheduler = torch.optim.lr_scheduler.StepLR(student_net_optimizer
                                                        , step_size=10, gamma=0.1)  # 设置学习率下降策略

# optimizer for transferBridge
transferBridge_optimizer_set = list()
transferBridge_scheduler_set = list()
for i in range(len(featureSet_list)):
    transferBridge_optimizer_set.append(optim.Adam(transferBridge_set[i].parameters(), lr=transforms_bridge_RL
                                                   , betas=(beta1, 0.999),  weight_decay=trans_bridge_weight_decay))
    transferBridge_scheduler_set.append(torch.optim.lr_scheduler.StepLR(transferBridge_optimizer_set[i], step_size=10, gamma=0.1))  # 设置学习率下降策略

# optimizer for common representation
common_representation_optimizer = optim.Adam([common_representation]
                                             , lr=LR, betas=(beta1, 0.999),  weight_decay=weight_decay)
common_representation_scheduler = torch.optim.lr_scheduler.StepLR(common_representation_optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

print("# ============================ step 5/6 recording ============================")
# 记录总迭代次数,方便tensorboard记录
transferBridge_iter_count = [0, 0, 0, 0, 0]
student_iter_count = 0
common_representation_iter_count = 0

comment = "weight_decay_train_STUDENT"+"_MAX-EPOCH:"+str(MAX_EPOCH)+"_LR:"+str(LR) + "_COM-REP-SIZE:" + str(COMMON_REPRESENTATION_SIZE)
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
        transferBridge_set[i].train()
        for batch_idx, data in enumerate(dataloader_set[i]):
            _, _, indexes = data
            labels = hidden_layer_set[i][indexes]
            outputs = transferBridge_set[i](common_representation[indexes])
            # print("-----------------------hidden layer is:\n", labels.shape, labels)
            # print("-----------------------outputs is:\n", outputs.shape, outputs)

            transferBridge_optimizer_set[i].zero_grad()
            loss = transferBridge_criterion(outputs, labels)
            loss.backward()
            transferBridge_optimizer_set[i].step()

            if batch_idx%10 == 0:
                print(loss.item())

        transferBridge_iter_count[i] += 1
        writer.add_scalar("Train {} transfer_bridge".format(featureSet_list[i])
                          , loss.item(), transferBridge_iter_count[i])
        # transferBridge_scheduler_set[i].step()

    print("================== phase 2: training student model ==================")
    for batch_idx, data in enumerate(common_representation_dataloader):
        student_iter_count += 1

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
        writer.add_scalar("Accuracy_Train", correct / total, student_iter_count)

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

    other_featureSet_size = training_samples_size - featureSet_size
    weight = (featureSet_size + other_featureSet_size) / featureSet_size / other_featureSet_size

    common_representation_iter_count += 1
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
    print(contrastive_loss)
    writer.add_scalar("Train contrastive_loss", contrastive_loss.item(), common_representation_iter_count)

    backward_start = time.time()
    contrastive_loss.backward()
    backward_end = time.time()
    print("\nit takes {}s to backward".format(str(backward_end - backward_start)))
    common_representation_optimizer.step()

    # 由于需要其他phase的grad,所以在这里清零梯度
    common_representation_optimizer.zero_grad()
    # common_representation_scheduler.step()

    if test_flag and (epoch > epoch_begin_test):
        for i in range(len(featureSet_list)):  # 如果某一方没有该样本需要输入吗???
            transferBridge_set[i].eval()

        print("================== phase 4: test ==================")
        '''
            对于每批测试样本:
                hidden_layers:list(tensor)每个tensor包含batchsize个hidden_layer, labels, indexes
                1.随机生成TEST_BATCH_SIZE的test_common_representation,及其test_optimizer
                2.TRAINING:
                    2.1.test_common_representation通过第transferBridge_set[i]得到outputs[i]并与对于的hidden_layyer计算MSELoss
                    2.2.累加MSELoss,backward,step
        '''
        correct = 0.
        total = 0.
        print_count = 1
        iter_count = 0

        # TODO(先生成序号,再通过序号获得,方便处理样本不完全的情况)
        for batch_idx, data in enumerate(test_dataloader):
            hidden_layers, labels, indexes = data

            if print_count:
                print_count = 0
                print("hidden_layers is \n", hidden_layers)
                for i in range(len(featureSet_list)):
                    print(hidden_layers[i].shape)
                print("labels is \n", labels)


            hidden_layers, labels, indexes = [sample.to(device) for sample in hidden_layers], labels.to(device), indexes.to(device)

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
                if converge % 10 == 0:
                    print("test loss: ", test_loss)

                iter_count += 1
                # 由于是从第101个epoch开始测试所以tensorboard上从404k开始记录
                writer.add_scalar("Test Loss", test_loss, iter_count+epoch*TEST_TRAINING_EPOCH*10)# TEST_TRAINING_EPOCH+10方便观察


            # 清空transferBridge的梯度
            print("========== clean  transferBridge's grad ==========")
            for i in range(len(featureSet_list)):
                # print(transferBridge_set[i].fc1.weight.grad)
                transferBridge_optimizer_set[i].zero_grad()
                # print(transferBridge_set[i].fc1.weight.grad)

            # prediction = student_net(共同表示) - 共同表示与各个类的contrastive_loss
            # !!!注意共同表示与各个类的contrastive_loss为负,所以需要减
            print("========== phase 4.2: predict ==========")
            pre_socre = student_net(test_common_representation)
            pre_socre = tensor_normalize(pre_socre)
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
                pre_socre[i] -= norm_contrastive_loss_set
                print("the contrastive ", norm_contrastive_loss_set)
                print("student predict score and contrastive_loss: ", pre_socre[i])

            _, predicted = torch.max(pre_socre.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()
        writer.add_scalar("Accuracy_Test", correct / total, epoch)
        print(" Loss: {:.4f} test_Acc:{:.2%}".format(test_loss, correct / total))


    if ((epoch + 1) % save_interval == 0) and save_flag:
        print("================== phase 5: save models ==================")
        # save transferBridge model
        for i in range(len(featureSet_list)):
            model_saved_path = "../saved_model/transferBridge_to_"\
                               + featureSet_list[i] + str(epoch) + ".pkl"
            torch.save(transferBridge_set[i].state_dict(), model_saved_path)

        # save student model
        model_saved_path = "../saved_model/student_net" + str(epoch) + ".pkl"
        torch.save(student_net.state_dict(), model_saved_path)

        # save common_representation
        np_common_representation = common_representation.detach().cpu().numpy()
        model_saved_path = "../saved_model/np_common_representation" + str(epoch) + ".npy"
        np.save(model_saved_path, np_common_representation)
