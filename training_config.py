import torch,os,sys,datetime,time
from tqdm import tqdm
import torch.optim as optim

best_acc=0

def train(net,trainloader,optim,criterion,epoch,device):
    print("Training")
    net.train()
    train_loss = 0
    total = 0
    total_correct = 0
    
    iterator = tqdm(trainloader)
    
    for inputs,targets in iterator:
        
        inputs,targets = inputs.to(device), targets.to(device)
        
        optim.zero_grad()
        outputs,_ = net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    print("Epoch: [{}]  loss: [{:.2f}] Accuracy [{:.2f}] ".format(epoch+1,train_loss/len(trainloader),
                                                                           total_correct*100/total))
    
def test(net,testloader,optim,criterion,epoch,device):
    global best_acc
    print("validation")
    net.eval()
    test_loss,total,total_correct = 0,0,0
    
    iterator = tqdm(testloader)
    
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs,_ = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        total_correct += (predicted == targets).sum().item()

    # Save checkpoint when best model
    acc = 100. * total_correct / total
    print("\nValidation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch+1, test_loss, acc))

    if acc > best_acc:
        print('Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+'model.t7')
        best_acc = acc
        
    return best_acc