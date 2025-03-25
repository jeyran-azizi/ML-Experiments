import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and construct dataset
training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

# Split training set into train and validation
training_size = int(0.8 * len(training_data))
validation_size = len(training_data) - training_size
training_data, validation_data = random_split(training_data, [training_size, validation_size])

# Data loaders
train_loader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=2)
validation_loader = DataLoader(validation_data, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
