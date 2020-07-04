from attacks import *
from tqdm import tqdm

def FGSM_test(net,testloader,optim,criterion,eps,device):
    print("\nFGSM Attack")
    net.eval()
    test_loss,total,total_correct = 0,0,0
    iterator = tqdm(testloader)
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        
        adv_inputs = FGSM(net,inputs,targets,device,eps,criterion)
        
        outputs,_ = net(adv_inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        total_correct += (predicted == targets).sum().item()

    # Save checkpoint when best model
    acc = 100. * total_correct / total
    print("FGSM Attack Acc Score \tLoss: %.4f Acc@1: %.2f%%" %(test_loss, acc))
    
    
def MIFGSM_test(net,testloader,optim,criterion,eps,max_iter,momentum,device):
    print("\nMIFGSM Attack")
    net.eval()
    test_loss,total,total_correct = 0,0,0
    iterator = tqdm(testloader)
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        
        adv_inputs = MIFGSM(net,inputs,targets,device,eps,max_iter,momentum,criterion)
        
        outputs,_ = net(adv_inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        total_correct += (predicted == targets).sum().item()

    # Save checkpoint when best model
    acc = 100. * total_correct / total
    print("MIFGSM Attack Acc Score \tLoss: %.4f Acc@1: %.2f%%" %(test_loss, acc))
    
    
    
def PGD_test(net,testloader,optim,criterion,eps,attack_steps,step_size,device):
    print("\nPGD Attack")
    net.eval()
    test_loss,total,total_correct = 0,0,0
    iterator = tqdm(testloader)
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        
        #adv_inputs = PGD(net,inputs,targets,device,eps,attack_steps,step_size,criterion)
        adv_inputs = PGD(net,inputs,targets,device,eps,attack_steps,step_size,criterion)        
        outputs,_ = net(adv_inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        total_correct += (predicted == targets).sum().item()

    # Save checkpoint when best model
    acc = 100. * total_correct / total
    print("PGD Attack Acc Score \tLoss: %.4f Acc@1: %.2f%%" %(test_loss, acc))
    
    
def FGSM_train(net,trainloader,optim,criterion,epoch,eps,device):
    net.train()
    train_loss, total, total_correct = 0,0,0
    
    iterator = tqdm(trainloader)
    
    for inputs,targets in iterator:
        
        net,inputs,targets = net.to(device),inputs.to(device), targets.to(device)
        
        adv_inputs = FGSM(net,inputs,targets,device,eps,criterion)
        
        optim.zero_grad()
        outputs,_ = net(adv_inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    print("Epoch: [{}]  loss: [{:.2f}] Accuracy [{:.2f}] ".format(epoch+1,train_loss/len(trainloader),
                                                                           total_correct*100/total))
    
    
    
def PGD_train(net,trainloader,optim,criterion,epoch,eps,attack_steps,step_size,device):
    net.train()
    train_loss, total, total_correct = 0,0,0
    
    iterator = tqdm(trainloader)
    
    for inputs,targets in iterator:
        
        net,inputs,targets = net.to(device),inputs.to(device), targets.to(device)
        
        adv_inputs = PGD(net,inputs,targets,device,eps,attack_steps,step_size,criterion)
        
        optim.zero_grad()
        outputs,_ = net(adv_inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    print("Epoch: [{}]  loss: [{:.2f}] Accuracy [{:.2f}] ".format(epoch+1,train_loss/len(trainloader),
                                                                           total_correct*100/total))
    
    
def FeaScatter_train(net,trainloader,optim,criterion,epoch,eps,attack_steps,step_size,device):
    net.train()
    train_loss, total, total_correct = 0,0,0
    
    iterator = tqdm(trainloader)
    
    for inputs,targets in iterator:
        
        net,inputs,targets = net.to(device),inputs.to(device), targets.to(device)
        
        optim.zero_grad()
        outputs,loss_fs = FeatureScatter(net,inputs,targets,device,eps,attack_steps,step_size,criterion)
        loss = loss_fs.mean()
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    print("Epoch: [{}]  loss: [{:.2f}] Accuracy [{:.2f}] ".format(epoch+1,train_loss/len(trainloader),
                                                                           total_correct*100/total))    