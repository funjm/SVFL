import torch
from torch.utils.data import Dataset
from tools.datasets import test_Dataset

featureSet_list = ['mfeat-fou', 'mfeat-fac', 'mfeat-kar', 'mfeat-pix', 'mfeat-zer']
testdataset_list = ['../datasets/Handwritten/mfeat/splite/' + file_name + '/test_data.npy' for file_name in featureSet_list]

testdataset = test_Dataset(testdataset_list)

test_dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=10, shuffle=False)

count = 3
for batch_idx, data in enumerate(test_dataloader):
    print("-------------data is :")
    inputs, labels, indexes = data
    print(inputs[0].shape)
    print(inputs[2].shape)
    print(inputs[0])
    print(labels)
    if batch_idx > 12:
        break