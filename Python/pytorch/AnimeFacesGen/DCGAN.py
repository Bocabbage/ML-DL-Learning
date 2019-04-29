import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os

#Hyper parameters
IS_CUDA = True
NGPU = 1
DATASET_PATH = 'E:/Programming/Dataset/AnimeFaces/'
IMAGE_SIZE = 64
BATCH_SIZE = 50
LR_G = 0.0001
LR_D = 0.0001
EPOCH = 1


# Load Dataset
dataset = torchvision.datasets.ImageFolder(root=DATASET_PATH,
                                           transform=transforms.Compose([
                                           transforms.Resize(IMAGE_SIZE),
                                           transforms.CenterCrop(IMAGE_SIZE),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

# data check
def imshow(img):
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

data_iter = iter(data_loader)
images, _ = data_iter.next()
print(images.numpy()[0].shape)
imshow(torchvision.utils.make_grid(images))

# Module Define
class Generator(nn.module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.fc = nn.Linear(100, 1024*4*4)
        self._main = nn.Sequential(
            # size: 1024*4*4
            nn.ConvTranspose2d(in_channels=1024,
                               out_channels=512, 
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # size: 512*8*8
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256, 
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # size: 256*16*16
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # size: 128*32*32
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=3, 
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
            # size: 3*64*64
        )

    def forward(self,input):
        inputs = self.fc(inputs)
        inputs = inputs.view(-1,1024,4,4)
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self._main,
                                               inputs,
                                               range(self.ngpu))
        else:
            output = self._main(input)

        return output


class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self._main = nn.Sequential(
            # input size: 3*64*64
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            # size: 64*32*32
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            # size: 128*16*16
            nn.Conv2d(in_channels=128, 
                      out_channels=256, 
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            # size: 256*8*8
            nn.Conv2d(in_channels=256, 
                      out_channels=512, 
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            # size: 512*4*4
            nn.Conv2d(in_channels=512,
                      out_channels=1, 
                      kernel_size=4,
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self,input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self._main,
                                               inputs,
                                               range(self.ngpu))
        else:
            output = self._main(input)

        return ouput.view(-1,1).squeeze(1)

