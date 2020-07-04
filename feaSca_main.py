import os,argparse,sys,datetime,time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from models import *
from attack_methods import *
from ot import *
from training_config import *
from utils import *
from use_attacks import *
from attacks import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



trainset = datasets.MNIST(root='/home/aminul/data', train = True, download =False, transform = transforms.ToTensor())
testset = datasets.MNIST(root='/home/aminul/data', train = False, download =False, transform = transforms.ToTensor())
num_classes = 10


train_loader = DataLoader(trainset, 100, shuffle =True, num_workers = 2)
test_loader = DataLoader(testset, 100, shuffle = False, num_workers = 2)


net = WideResNet(depth=28,
                           num_classes=num_classes,
                           widen_factor=10)
net = net.to(device)
net = nn.DataParallel(net, device_ids=[0, 1, 2])


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.1)


for epoch in range(0, 30):
    FeaScatter_train(net,train_loader,optimizer,criterion,epoch,0.15,1,0.008,device)
    best_acc = test(net,test_loader,optimizer,criterion,epoch,device)


print("\n\n\n----------------")
print("Loading Best Model")
checkpoint = torch.load('./checkpoint/model.t7')
net = checkpoint['net']
print("Done !")

FGSM_test(net,test_loader,optimizer,criterion,0.15,device)    
PGD_test(net,test_loader,optimizer,criterion,0.15,10,0.08,device)

