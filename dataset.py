import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
class DATASET():
        def MNIST():
            train_dataset = torchvision.datasets.MNIST(root='../../data',
                                                    train=True,
                                                    transform=transforms.ToTensor(),download=True)

            test_dataset = torchvision.datasets.MNIST(root='../../data',
                                                    train=False,
                                                    transform=transforms.ToTensor())
            return train_dataset, test_dataset

        def CIFAR10():
            train_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                                    train=True,
                                                    transform=transforms.ToTensor(),download=True)

            test_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                                    train=False,
                                                    transform=transforms.ToTensor())


            return train_dataset, test_dataset

        def dataset3():
            
            return train_dataset, test_dataset
