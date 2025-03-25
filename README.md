# Machine Learning with PyTorch

This repository contains a collection of machine learning projects implemented with **PyTorch**. It includes various models and experiments related to **image classification**, **deep learning**, and **machine learning** algorithms. Each folder corresponds to a different project, and the projects are designed to explore various techniques in the field of machine learning, primarily focused on computer vision tasks.

## Projects Included

- **CIFAR-10 Classification with CNN**: A Convolutional Neural Network (CNN) implementation for classifying CIFAR-10 dataset images.
- **ResNet for CIFAR-10**: A ResNet model implementation applied to the CIFAR-10 dataset.

More projects will be added over time, covering different machine learning techniques and use cases.

## Installation

### Clone this repository:

```bash
git clone https://github.com/jeyran-azizi/ML-Experiments.git
cd ML-Experiments
```

### Install dependencies:

You can install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the dependencies with the following command:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

### Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- numpy

## Usage

### 1. Training a model

To train a model, follow these steps:

- Navigate to the appropriate project folder (e.g., `cnn` or `resnet`).
- Run the training script. For example, to train the CNN model:

```bash
python train.py
```

Make sure to modify the scripts as per your dataset or training configurations.

### 2. Running the Model

Once the model is trained, you can test it using:

```bash
python test.py
```

The test script will evaluate the model's performance on the test data and output various metrics (accuracy, precision, recall, F1-score).


## Acknowledgements

- CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch (https://pytorch.org/)
- torchvision (https://pytorch.org/vision/stable/index.html)
