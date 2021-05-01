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
from model.MLP import TeacherMLP
from tools.datasets import handWritten_Dataset

# data parameters setting
CLASS = 10

# training parameters setting
MAX_EPOCH = 50
BATCH_SIZE = 40
LR = 0.01
# loss记录间隔
log_interval = 10
val_interval = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# file_name = 'mfeat-fou'#train:94%;test:86%#no_trick:97/82
# file_name = 'mfeat-fac'#train:99%;test:98%#no_trick:100/97
# file_name = 'mfeat-kar'#train:99%;test:96%#no_trick:100/96
# file_name = 'mfeat-pix'#train:99%;test:98%#no_trick:100/97
file_name = 'mfeat-zer'#train:90%;test:87%#no_trick:90/85

# # 效果不好先不用
# file_name = 'mfeat-mor'#train:76%;test:75%#no_trick:

train_file_path = '../datasets/Handwritten/mfeat/splite/' + file_name + '/train_data.npy'
test_file_path = '../datasets/Handwritten/mfeat/splite/' + file_name + '/test_data.npy'

model_saved_path = "../saved_model/"+file_name+".pkl"
# ============================ step 1/5 data ============================
# norm_mean = [0.18554415, 0.3773108, 0.26937738, 0.26568627, 0.3003665, 0.15831402, 0.36477035, 0.24721313, 0.2886569, 0.15171565, 0.15711898, 0.1665694, 0.16424574, 0.19058165, 0.1424347, 0.11498316, 0.1276785, 0.13093743, 0.11939616, 0.11682219, 0.10725221, 0.10872045, 0.09840176, 0.11451166, 0.10574758, 0.09893811, 0.0950002, 0.09061539, 0.092172936, 0.09469713, 0.09356688, 0.08721957, 0.0824841, 0.08297376, 0.09123025, 0.086685814, 0.08164547, 0.07657155, 0.077050135, 0.08538279, 0.081619024, 0.07713094, 0.073311865, 0.07169203, 0.08380526, 0.078134395, 0.07440393, 0.07265592, 0.08630855, 0.08285595, 0.08020027, 0.07914363, 0.09354703, 0.091153584, 0.086378925, 0.0854833, 0.10129984, 0.100857325, 0.090473846, 0.09434485, 0.11591607, 0.10952404, 0.104976125, 0.10964255, 0.11769509, 0.13126543, 0.12350266, 0.12840894, 0.15797728, 0.18296516, 0.14995685, 0.1667022, 0.23095599, 0.26778576, 0.14398222, 0.21779197]
# norm_std = [0.09187492, 0.1761866, 0.13794588, 0.11066662, 0.1562168, 0.088999614, 0.14433108, 0.123825386, 0.11588295, 0.08720216, 0.08632226, 0.08277992, 0.09120054, 0.11025743, 0.06420329, 0.058655243, 0.06574285, 0.07804482, 0.06492451, 0.05284898, 0.051824175, 0.052133236, 0.057535563, 0.06370528, 0.046820886, 0.04965517, 0.04701156, 0.049862903, 0.0517168, 0.044781767, 0.045007885, 0.042378668, 0.04625211, 0.04405012, 0.0424639, 0.042378016, 0.04081059, 0.04123131, 0.041325, 0.040798236, 0.03954191, 0.038189203, 0.038312733, 0.038576484, 0.04066802, 0.03754611, 0.038965248, 0.037853625, 0.04215067, 0.041119948, 0.042057272, 0.04202717, 0.044471443, 0.044336542, 0.047646385, 0.043382812, 0.050210826, 0.04962741, 0.048542015, 0.052643232, 0.05951389, 0.05343499, 0.060597982, 0.055932388, 0.061979853, 0.066582456, 0.06880583, 0.06478485, 0.090917245, 0.086287655, 0.08824494, 0.09121716, 0.12341024, 0.13097644, 0.0856414, 0.12287771]
#
# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(norm_mean, norm_std),
# ])
#
# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(norm_mean, norm_std),
# ])

train_data = handWritten_Dataset(data_path=train_file_path)
test_data = handWritten_Dataset(data_path=test_file_path, train=False)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
# ============================ step 2/5 model ============================
print("the feature size is {}".format(train_data.get_features_size()))
net = TeacherMLP(classes=CLASS, input_size=train_data.get_features_size())
net.to(device)
net.initialize_weights()

# ============================ step 3/5 loss function ============================
criterion = nn.CrossEntropyLoss()

# ============================ step 4/5 optimizer ============================

# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-2)  # L2
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

# ============================ step 5/6 hook ============================
# 使用hook记录中间层输出
hidden_block = list()

def forward_hook(module, data_input, data_output):
    # print(data_output.shape)
    hidden_block.append(data_output)

# net.fc1.register_forward_hook(forward_hook)
list(net.modules())[1].register_forward_hook(forward_hook)

# 添加保持中间变量的网络结构,每次不仅算output,还返回中间结果
net_hidden1 = list()
net_hidden2 = list()

# ============================ step 6/6 training ============================
train_curve = list()
valid_curve = list()

# 记录总迭代次数,方便tensorboard记录
iter_count = 0
comment = " _"+file_name+"_MAX_EPOCH:"+str(MAX_EPOCH)+"_LR:"+str(LR)+"_BATCHSIZE:"+str(BATCH_SIZE)
# writer = SummaryWriter(comment='', log_dir='../runs', filename_suffix=filename_suffix)
filename_name = comment
# writer = SummaryWriter(comment=comment, filename_suffix=filename_name, log_dir='no_trick_MLP')
writer = SummaryWriter(comment=comment, filename_suffix=filename_name)

for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    for batch_idx, data in enumerate(train_loader):
        iter_count += 1

        inputs, labels, indexes = data
        inputs, labels, indexes = inputs.to(device), labels.to(device), indexes.to(device)

        outputs, tmp_h1, tmp_h2 = net(inputs)
        net_hidden1.append(tmp_h1)
        net_hidden2.append(tmp_h2)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()
        # optimizer.zero_grad()   #zero_grad放在step后也是可行的


        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).cpu().squeeze().sum().numpy()


        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (batch_idx+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, batch_idx+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

        writer.add_scalar("Loss_Train", loss.item(), iter_count)
        # writer.add_scalars("Loss_Train", loss, iter_count)
        writer.add_scalar("Accuracy_Train", correct / total, iter_count)

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                inputs, labels, indexes = data
                inputs, labels, indexes = inputs.to(device), labels.to(device), indexes.to(device)

                outputs, tmp_h1, tmp_h2 = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).cpu().squeeze().sum().numpy()

                loss_val += loss.item()

            loss_val_epoch = loss_val / len(test_loader)
            valid_curve.append(loss_val_epoch)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(test_loader), loss_val_epoch, correct_val / total_val))

            writer.add_scalar("Loss_Valid", loss_val_epoch, iter_count)
            writer.add_scalar("Accuracy_Valid", correct_val / total_val, iter_count)


train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval - 1  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

# save model
torch.save(net.state_dict(), model_saved_path)

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()


# print("feature maps shape: {}\noutput value: {}\n".format(fmap_block[0].shape, fmap_block[0]))
# print("input shape: {}\ninput value: {}".format(input_block[0][0].shape, input_block[0]))

print("finish")