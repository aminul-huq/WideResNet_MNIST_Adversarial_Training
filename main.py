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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MNIST')

    # model hyper-parameter variables
    parser.add_argument('--itr', default=100, metavar='itr', type=int, help='no. of iteration for training')
    parser.add_argument('--eps', default=0.15, metavar='eps', type=float, help='epsilon value')
    parser.add_argument('--max_iter', default=10, metavar='max_iter', type=int, help='maximum iteration for attack convergence')
    parser.add_argument('--step_size', default=0.08, metavar='step_size', type=float, help='step size in each iteration of attack')
    
    args = parser.parse_args()


num_epochs = args.itr
eps = args.eps
max_iter = args.max_iter
step_size = args.step_size

trainset = datasets.MNIST(root='/home/aminul/data', train = True, download =False, transform = transforms.ToTensor())
testset = datasets.MNIST(root='/home/aminul/data', train = False, download =False, transform = transforms.ToTensor())
num_classes = 10


train_loader = DataLoader(trainset, 100, shuffle =True, num_workers = 2)
test_loader = DataLoader(testset, 100, shuffle = False, num_workers = 2)


net = WideResNet(depth=28,
                           num_classes=num_classes,
                           widen_factor=10)
net = net.to(device)
net = nn.DataParallel(net, device_ids=[1,2])


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.1)


print("Standard Training Performed on Original Data")
for epoch in range(0, num_epochs):
    
    train(net,train_loader,optimizer,criterion,epoch,device)
    best_acc = test(net,test_loader,optimizer,criterion,epoch,device,'standard_')

print("\n\n\n----------------")
print("Loading Best Model")
checkpoint = torch.load('./checkpoint/standard_model.t7')
net = checkpoint['net']
print("Done !")

print("\nRobustness Checking on Standard Training on Different Attacks")
FGSM_test(net,test_loader,optimizer,criterion,eps,device)    
PGD_test(net,test_loader,optimizer,criterion,eps,max_iter,step_size,device)

print("\n\nAdversarial Training Performed Using PGD")

for epoch in range(0, num_epochs):
    PGD_train(net,train_loader,optimizer,criterion,epoch,eps,max_iter,step_size,device)
    best_acc = test(net,test_loader,optimizer,criterion,epoch,device,'PGD_adversarial_')

print("\n\n\n----------------")
print("Loading Best Model")
checkpoint = torch.load('./checkpoint/PGD_adversarial_model.t7')
net = checkpoint['net']
print("Done !")    
    
FGSM_test(net,test_loader,optimizer,criterion,eps,device)    
PGD_test(net,test_loader,optimizer,criterion,eps,max_iter,step_size,device)
