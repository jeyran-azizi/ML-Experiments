# ResNet Implementation on CIFAR-10

This repository contains an implementation of ResNet on the CIFAR-10 dataset using PyTorch.

## Features
- Uses a ResNet architecture with residual blocks
- Trains on the CIFAR-10 dataset
- Performs validation during training
- Saves the trained model for future inference

## Installation

### Clone this repository:
```sh
git clone https://github.com/your-username/cifar10-pytorch.git
cd cifar10-pytorch/resnet
```

### Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

### Train the model:
Run the following command to train the ResNet model:
```sh
python train.py
```

### Evaluate the model:
To test the trained model, run:
```sh
python evaluate.py
```

## Dataset
The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.

## Results
After training, the model achieves 79% accuracy on the CIFAR-10 validation set.

## Folder Structure
```
resnet/
│── models.py          # Defines the ResNet model
│── train.py           # Training script
│── evaluate.py        # Evaluation script
│── utils.py           # Helper functions
│── data_loader.py     # Data preprocessing and loading
```

## Acknowledgments
This implementation is based on standard ResNet architectures and PyTorch best practices.
