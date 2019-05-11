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
import argparse

# parser = argparse.ArgumentParser()
parser.add_argument('--epoch',type=int,default=10)
parser.add_argument('--ngpu',type=int,default=0)
parser.add_argument('--dpath',default='./')
parser.add_argument('--opipath',default='./')
parser.add_argument('--opapath',default='./')
parser.add_argument('--netG',default='')
parser.add_argument('--netD',default='')
opt = parser.parse_args()

# Default Hyper parameters
# OUTPIC_PATH="/home/ljw/AnimeFacesGen/out_pic"
# OUTPARA_PATH="/home/ljw/AnimeFacesGen/out_para"
# DATASET_PATH="/home/ljw/AnimeFacesGen/AnimeFaces"
# NGPU = 1
# EPOCH = 100
OUTPIC_PATH = opt.opipath
OUTPARA_PATH = opt.opapath
DATASET_PATH = opt.dpath
NGPU = opt.ngpu
EPOCH = opt.epoch
PARA_G = ""
PARA_D = ""

NZ = 100
IMAGE_SIZE = 64
BATCH_SIZE = 50
LR_G = 0.0002
LR_D = 0.0002
BETAS = (0.5,0.999)


# GPU Setting
#NGPU = 0
if torch.cuda.is_available() and NGPU!=0:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


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
    img = img/2 + 0.5
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

data_iter = iter(data_loader)
images, _ = data_iter.next()
print(images.numpy()[0].shape)
imshow(torchvision.utils.make_grid(images))

# Module Define (DCGAN)
class Generator(nn.Module):
    def __init__(self,ngpu=0):
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

    def forward(self,inputs):
        inputs = self.fc(inputs)
        inputs = inputs.view(-1,1024,4,4)
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self._main,
                                               inputs,
                                               range(self.ngpu))
        else:
            output = self._main(inputs)

        return output


class Discriminator(nn.Module):
    def __init__(self,ngpu=0):
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

        return output.view(-1,1).squeeze(1)

# Random weight initialized 
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

print(device)

netG = Generator(NGPU).to(device)
netG.apply(weight_init)
if PARA_G != '':
    netG.load_state_dict(torch.load(PARA_G))
#print(netG)

netD = Discriminator(NGPU).to(device)
netD.apply(weight_init)
if PARA_D != '':
    netD.load_state_dict(torch.load(PARA_G))
#print(netD)

# Criterion and Optimizer options
criterion = nn.BCELoss()
real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(),lr=LR_D,betas=BETAS)
optimizerG = optim.Adam(netG.parameters(),lr=LR_G,betas=BETAS)

for epoch in range(EPOCH):
    for batch_idx,data in enumerate(data_loader):
        ###### Step1: Update Discriminator ######
        ### Maximize log(D(x))+log(1-D(G(z))) ###
        #########################################
        # train with real #
        netD.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        labels = torch.full((batch_size,),real_label,device=device)

        output = netD(real_data)
        errD_real = criterion(output,labels)
        errD_real.backward()

        # train with fake #
        noise = torch.randn(batch_size,NZ,device=device)
        fake = netG(noise)
        # replace the 'labels' with fake_label in-place
        labels.fill_(fake_label)
        # detach it so it won't influence the parameters in G
        output = netD(fake.detach())
        errD_fake = criterion(output,labels)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        #########################################

        ###### Step2: Update Generator ######
        ###### Maximize log(D(G(z)))   ######
        #####################################
        netG.zero_grad()
        labels.fill_(real_label)
        output = netD(fake)
        errG = criterion(output,labels)
        errG.backward()

        optimizerG.step()

        #####################################
        if batch_idx % 50 == 0:
            print('[%d/%d][%d/%d] Loss_D:%.4f | Loss_G:%.4f'
                  %(epoch,EPOCH,batch_idx,len(data_loader),
                    errD.item(),errG.item()))

    torchvision.utils.save_image(fake.detach(), 
                                 '%s/fake_samples_epoch_%03d.jpg'%
                                 (OUTPIC_PATH,epoch))
    torchvision.utils.save_image(real_data.detach(),'%s/real_samples_epoch_%03d.jpg'%
                                 (OUTPIC_PATH,epoch))

    # Save Para
    torch.save(netG.state_dict(),'%s/netG_epoch_%d.pth'%(OUTPARA_PATH,epoch))
    torch.save(netD.state_dict(),'%s/netD_epoch_%d.pth'%(OUTPARA_PATH,epoch))




