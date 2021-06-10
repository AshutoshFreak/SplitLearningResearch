import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

class DATALOADER:
    def train_dataloader(num_clients, train_dataset):
        train_batch_size = len(train_dataset)//num_clients
        train_split = [train_batch_size]*(num_clients-1)
        train_split.append(len(train_dataset)-train_batch_size*(num_clients-1))

        train_datasets = list(torch.utils.data.random_split(train_dataset, train_split))

        train_loaders = []
        for train_dataset in train_datasets:
            train_loaders.append(torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=train_batch_size,
                                                        shuffle=True))
        
        return train_loaders

    def test_dataloader(num_clients, test_dataset):
        test_batch_size = len(test_dataset)//num_clients
        test_split = [test_batch_size]*(num_clients-1)
        test_split.append(len(test_dataset)-test_batch_size*(num_clients-1))

        test_datasets = list(torch.utils.data.random_split(test_dataset, test_split))

        test_loaders = []
        for test_dataset in test_datasets:
            test_loaders.append(torch.utils.data.DataLoader(dataset=test_dataset,
                                                        batch_size=test_batch_size,
                                                        shuffle=True))
        return test_loaders