import torch
from ot import * 
from utils import *

def FGSM(net,inputs,labels,device,eps,criterion):
    
    net,inputs,labels = net.to(device),inputs.to(device),labels.to(device)
       
    x = inputs.clone().detach().requires_grad_(True).to(device)
    alpha = eps 

    pred,_ = net(x)
    loss = criterion(pred,labels)
    loss.backward()
    noise = x.grad.data

    x.data = x.data + alpha * torch.sign(noise)
    x.data.clamp_(min=0.0, max=1.0)
    x.grad.zero_()
    
    return x



def PGD(net,inputs,labels,device,eps,iters,alpha,criterion):
    
        inputs,labels = inputs.to(device),labels.to(device)
        ori_images = inputs.clone().detach()
                
        inputs = inputs + torch.empty_like(inputs).uniform_(-eps, eps)
        inputs = torch.clamp(inputs, min=0, max=1)
        
        for i in range(iters) :    
            inputs.requires_grad = True
            outputs,_ = net(inputs)
            
            cost = criterion(outputs, labels).to(device)
            
            grad = torch.autograd.grad(cost, inputs, 
                                       retain_graph=False, create_graph=False)[0]

            adv_images = inputs + alpha*grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            inputs = torch.clamp(ori_images + eta, min=0, max=1).detach()

        adv_images = inputs
        
        return adv_images  


def MIFGSM(net,inputs,labels,device,eps,max_iter,momentum,criterion):
    
    alpha = eps/max_iter
    
    net,inputs,labels = net.to(device),inputs.to(device),labels.to(device)

    x_adv = inputs.clone().detach().requires_grad_(True).to(device)
       
    g = torch.zeros(inputs.size(0), 1, 1).to(device)
    
    for i in range(max_iter):
        pred,_ = net(x_adv)
        loss = criterion(pred,labels)
        loss.backward()
        
        x_grad = x_adv.grad.data 
        
        g = momentum * g.data + x_grad/x_grad.abs().mean(dim=(1, 2), keepdim=True)
        
        x_adv.data = x_adv.data + alpha * torch.sign(g)
        
        x_adv.data.clamp_(min=0.0, max=1.0)
    
    return x_adv



def FeatureScatter(net,inputs,labels,device,eps,iters,alpha,criterion):

    batch_size = inputs.size(0)
    m,n = batch_size,batch_size
    
    logits = net(inputs)[0]
    num_classes = logits.size(1)
    outputs = net(inputs)[0]
    
    targets_prob = F.softmax(outputs.float(), dim=1)
    y_tensor_adv = labels
    
    x = inputs.detach()
    x_org = x.detach()
    
    x = x + torch.zeros_like(x).uniform_(-eps,eps)

    logits_pred_nat, fea_nat = net(inputs)
    
    y_gt = one_hot_tensor(labels, num_classes, device)

    loss_ce = softCrossEntropy()

    iter_num = iters # no. of iterations or max_iters
    
    
    
    for i in range(iter_num):
            x.requires_grad_()
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = net(x)

            ot_loss = sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n)

            net.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + alpha * torch.sign(x.grad.data)  # step_size might be alpha value
            x_adv = torch.min(torch.max(x_adv, inputs - eps),
                              inputs + eps)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)  # might be 0 to 1
            x = Variable(x_adv)

            logits_pred, fea = net(x)
            net.zero_grad()

            y_sm = label_smoothing(y_gt, y_gt.size(1), 0.5)  
            # according to main repo ls_factor = 0.5 for cifar10/100 and 0.1 if undefined

            adv_loss = loss_ce(logits_pred, y_sm.detach())

    return logits_pred, adv_loss

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    