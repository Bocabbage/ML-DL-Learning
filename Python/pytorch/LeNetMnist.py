import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Hyper Parameters
EPOCH = 1
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
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(x,self):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    #print(npimg.shape)
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

data_iter = iter(train_loader)
images,labels = data_iter.next()
imshow(torchvision.utils.make_grid(images))

