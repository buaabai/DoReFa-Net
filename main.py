import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import quantizek_util as Q

def main():
    
    BATCH_SIZE = 100
    TEST_BATCH_SIZE = 100
    learning_rate = 1e-3
    #momentum = args.momentum
    weight_decay = 1e-5

    ###################################################################
    ##             Load Train Dataset                                ##
    ###################################################################
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
    ###################################################################
    ##             Load Test Dataset                                ##
    ###################################################################
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, download=False,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=TEST_BATCH_SIZE, shuffle=True)
    model = Q.LeNet5_Q()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

    best_acc = 0.0 
    for epoch_index in range(1,100+1):
        adjust_learning_rate(learning_rate,optimizer,epoch_index,20)
        train(epoch_index,train_loader,model,optimizer,criterion)
        acc = test(model,test_loader,criterion)
        if acc > best_acc:
            best_acc = acc
            #save_model(model,best_acc)
 
def save_model(model,acc):
    print('==>>>Saving model ...')
    state = {
        'acc':acc,
        'state_dict':model.state_dict() 
    }
    torch.save(state,'model_state.pkl')
    print('*** DONE! ***')

def train(epoch_index,train_loader,model,optimizer,criterion):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = Variable(data),Variable(target)

        optimizer.zero_grad()


        output = model(data)
        loss = criterion(output,target)
        loss.backward()


        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model,test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0

    for data,target in test_loader:
        data,target = Variable(data),Variable(target)
        output = model(data)
        test_loss += criterion(output,target).item()
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    acc = 100. * correct/len(test_loader.dataset)

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return acc
    
def adjust_learning_rate(learning_rate,optimizer,epoch_index,lr_epoch):
    lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr

if __name__ == '__main__':
    main()