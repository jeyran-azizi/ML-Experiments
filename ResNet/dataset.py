import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

def get_dataloaders(batch_size=64, validation_size=0.1, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Splitting training data into training and validation sets
    indices = list(range(len(train_data)))
    split = int(np.floor(validation_size * len(train_data)))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(train_data, batch_size=batch_size, sampler=validation_sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader
