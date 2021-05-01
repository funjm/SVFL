import torch.nn as nn
import torch.nn.functional as F

FC1_SIZE = 200
FC2_SIZE = 100


class TeacherMLP(nn.Module):
    def __init__(self, classes, input_size):
        super(TeacherMLP, self).__init__()

        self.hidden1_output = None
        self.hidden2_output = None

        FC1_SIZE = 2*input_size
        print("FC1_SIZE is :{}\nFC2_SIZE is {}".format(FC1_SIZE, FC2_SIZE))

        # =======================无trick=======================
        self.fc1 = nn.Linear(input_size, FC1_SIZE)
        self.fc2 = nn.Linear(FC1_SIZE, FC2_SIZE)
        self.fc3 = nn.Linear(FC2_SIZE, classes)

        # # =======================tricks: dropout, BN=======================
        # self.fc1 = nn.Linear(input_size, FC1_SIZE)
        # self.bn1 = nn.BatchNorm1d(num_features=FC1_SIZE)
        #
        # self.dropout = nn.Dropout(p=0.3)
        #
        # self.fc2 = nn.Linear(FC1_SIZE, FC2_SIZE)
        # self.bn2 = nn.BatchNorm1d(num_features=FC2_SIZE)
        # self.fc3 = nn.Linear(FC2_SIZE, classes)



        # self.layers = nn.Sequential(
        #     nn.Linear(input_size, 200),
        #     nn.BatchNorm1d(num_features=200),
        #     nn.ReLU(),
        #     nn.Linear(200, 100),
        #     nn.BatchNorm1d(num_features=100),
        #     nn.ReLU(),
        #     nn.Linear(100, classes)
        # )

    def forward(self, x):
        '''Forward pass'''
        # =======================无trick=======================
        x = self.fc1(x)
        self.hidden1_output = x
        x = F.relu(x)

        x = self.fc2(x)
        self.hidden2_output = x
        x = F.relu(x)

        x = self.fc3(x)

        # # =======================tricks: dropout, BN=======================
        # x = self.fc1(x)
        # # 记录hidden1的输出
        # self.hidden1_output = x
        # x = self.bn1(x)
        # x = F.relu(x)
        #
        # x = self.fc2(x)
        # # 记录hidden1的输出
        # self.hidden2_output = x
        # x = self.bn2(x)
        # x = F.relu(x)
        #
        # x = self.dropout(x)
        #
        # x = self.fc3(x)

        return x, self.hidden1_output, self.hidden2_output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class StudentMLP(nn.Module):
    def __init__(self, classes, com_representation_size):
        super(StudentMLP, self).__init__()
        self.fc1 = nn.Linear(com_representation_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, classes)

    def forward(self, x):
        x = self.fc1(x)
        self.hidden1_output = x
        x = F.relu(x)

        x = self.fc2(x)
        self.hidden2_output = x
        x = F.relu(x)

        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class TransferBridge(nn.Module):
    def __init__(self, intermediate_size, com_representation_size):
        super(TransferBridge, self).__init__()
        # =======================无trick=======================
        self.fc1 = nn.Linear(com_representation_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, intermediate_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


