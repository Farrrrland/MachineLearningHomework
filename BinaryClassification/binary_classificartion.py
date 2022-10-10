#Don't change batch size
batch_size = 64

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import sys
import os

## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('./data/mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
subset_indices = ((train_data.train_labels == 0) + (train_data.train_labels == 1)).nonzero().reshape(-1)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))


subset_indices = ((test_data.test_labels == 0) + (test_data.test_labels == 1)).nonzero().reshape(-1)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))

# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.

# Hyper parameters
input_size = 784
num_classes = 1
num_epochs = 50
learning_rate = 0.001
momentums = [0, 0.8, 0.9, 0.95, 0.98]

# Model Definition
class LinearModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x):
        out = self.linear(x).squeeze()
        return out

class LogisticLoss(nn.Module):
    def __init__(self):
        super(LogisticLoss, self).__init__()
    def forward(self, inputs, target):
        return torch.mean(torch.log(1.0/torch.sigmoid(target * inputs)))
    
class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
    def forward(self, inputs, target):
        return torch.mean((1 - target * inputs).clamp(min=0))

def runModel(model, criterion, fname, momentum):
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
    with open(fname, 'w') as f:
        for epoch in range(num_epochs):
            total_loss = 0
            print("len of train data: " + str(len(train_data)))
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images.view(-1, 28 * 28))
                labels = Variable(2 * (labels.float() - 0.5))
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (i + 1) % 20 == 0:
                    print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                        % (epoch + 1, num_epochs, i + 1,
                            len(train_data) // batch_size, loss.item()))
            # Print your results every epoch
            epoch_loss = total_loss / len(train_loader)
            correct = 0
            total = 0
            for images, labels in test_loader:
                labels = Variable(2 * (labels.float() - 0.5))
                images = Variable(images.view(-1, 28 * 28))
                outputs = model(images)
                predicted = torch.where(torch.sigmoid(outputs)>0.5, 1, -1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            f.write('Epoch: [% d/% d]: Loss: %.4f , Accuracy of the currenent model: % .4f %%\n' % (
                epoch + 1, num_epochs,
                epoch_loss,
                100.0 * correct / total))

if __name__ == "__main__":
    # Logistic Regression
    for idx, momentum in enumerate(momentums):
        fname = sys.path[0] + os.sep + "log" + os.sep + f"epoch_logistic_regression_{idx}.txt"
        runModel(LinearModel(input_size, num_classes), LogisticLoss(), fname, momentum)
    # SVM
    for idx, momentum in enumerate(momentums):
        fname = sys.path[0] + os.sep + "log" + os.sep + f"epoch_svm_{idx}.txt"
        runModel(LinearModel(input_size, num_classes), HingeLoss(), fname, momentum)