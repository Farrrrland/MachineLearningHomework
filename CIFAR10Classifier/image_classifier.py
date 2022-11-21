import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import sys
import os


def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, train, test_loader

batch_size = 64
test_batch_size = 64
n_epochs = 20


train_loader, train_data, _ = cifar_loaders(batch_size)
_, _, test_loader = cifar_loaders(test_batch_size)

lossfunc = torch.nn.CrossEntropyLoss()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

class MLP_non_act(nn.Module):
    def __init__(self):
        super(MLP_non_act, self).__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return x

def runModel(model, fname):
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    with open(fname, 'w') as f:
        for epoch in range(1, n_epochs+1):
            total_loss = 0
            model.train()
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i%100 == 0:
                    print(epoch, i, loss.item())
            model.eval()
            correct = 0
            total = 0
            epoch_loss = total_loss / len(train_loader)
            for images, labels in test_loader:
                output = model(images)
                predicted = output.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            # print('Accuracy of the currenent model: % .4f %%\n' % (100.0 * correct / total))
            f.write('Epoch: [% d/% d]: Loss: %.4f , Accuracy of the currenent model: % .4f %%\n' % (
                    epoch, n_epochs,
                    epoch_loss,
                    100.0 * correct / total))

if __name__ == "__main__":
    fname = sys.path[0] + os.sep + "log" + os.sep + "mlp.txt"
    runModel(MLP(), fname)
    print("************** Separate Line **************")
    fname = sys.path[0] + os.sep + "log" + os.sep + "mlp_non_act.txt"
    runModel(MLP_non_act(), fname)
    print("************** Separate Line **************")
    fname = sys.path[0] + os.sep + "log" + os.sep + "cnn.txt"
    runModel(CNN(), fname)