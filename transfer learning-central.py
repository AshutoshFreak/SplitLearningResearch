import torch
import torchvision
from torchvision import models
from torchvision import transforms
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 50
num_epochs = 5

model = models.resnet18(pretrained=True)

# CIFAR10 dataset

# !wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# !tar -zxvf cifar-10-python.tar.gz ../../data

train_dataset = torchvision.datasets.CIFAR10(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


num_ftrs = model.fc.in_features
# print(num_ftrs)
print(model)
print()
print()
model.fc = nn.Linear(num_ftrs, 10)
children = list(model.children())
part1 = nn.Sequential(*children[:6])
part2 = nn.Sequential(*children[6:])

# Train the model
n_total_steps = len(train_loader)
train_acc = []
test_acc = []
for epoch in range(num_epochs):
    n_correct = 0
    n_samples = 0
    for i, (images, labels) in enumerate(train_loader):  
        # original shape: [100, 1, 28, 28]
        # resized: [100, 784]
        # print(images.shape)
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Calculating Train Accuracy
        if i == 0:
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    n_correct += (predicted == labels).sum().item()
                    n_samples += labels.size(0)
                train_acc.append(100*n_correct/n_samples)

            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
                test_acc.append(100.0 * n_correct / n_samples)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        if(i == 0):
            print(f'Train Accuracy: {train_acc[-1]}, Test Accuracy: {test_acc[-1]}')