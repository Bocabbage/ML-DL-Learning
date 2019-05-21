#!/usr/bin/env
# -*- coding: utf-8 -*- #
'''
    Python Version: Python3.7.2
    Update date: 2019/04/29(finished)
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time

# Hyper Parameters
EPOCH = 3
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

dataset_path = 'E:/Programming/Dataset'
if not (os.path.exists(dataset_path)):
    DOWNLOAD_MNIST = True

# Module define (LeNet-5) #
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        # Default: bias = True, stride = 1
        self.conv1 = nn.Conv2d(1, 6, 5)
        # kernal_size = 2, stride = 2
        # Default: padding = 0
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # pytorch: nn.CrossEntropyLoss() Doc:
        # the inputs are tensors that without being
        # processed with softmax
        # x = F.softmax(self.fc3(x))
        return x

net = LeNet5()



# MNIST load #
train_data = torchvision.datasets.MNIST(root=dataset_path,
                                        transform=torchvision.transforms.ToTensor(),
                                        train=True,download=DOWNLOAD_MNIST)
test_data = torchvision.datasets.MNIST(root=dataset_path,
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False,download=DOWNLOAD_MNIST)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
# [?]Add 'num_works' will bring an error

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)
# image visilization #
def imshow(img):
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

data_iter = iter(train_loader)
images,labels = data_iter.next()
imshow(torchvision.utils.make_grid(images))

# Training #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=LR)

start_time = time.time()
for epoch in range(EPOCH):
    for batch_idx,data in enumerate(train_loader):
        inputs,labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('[Epoch:%d,batch:%5d] loss:%.3f'%
                  (epoch+1,batch_idx,loss.data.numpy()))

print('Finish Traning in %.3f s'%(time.time()-start_time))

# Test #
correct = 0
total = 0

with torch.no_grad():
    # torch.no_grad
    # Context-manager that disabled gradient calculation.
    # It will reduce memory consumption for computations.
    for data in test_loader:
        inputs,labels = data
        outputs = net(inputs)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy: %.3f %%' % (100 * correct / total))
