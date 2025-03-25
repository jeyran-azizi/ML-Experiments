# Convolutional Neural Network (CNN) for CIFAR-10 Classification

This project implements a Convolutional Neural Network from scratch using PyTorch to classify images in the CIFAR-10 dataset.

**Project Overview:**
- Dataset: CIFAR-10 (10 classes, 60,000 images)
- Model: Custom CNN with two convolutional layers
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Metrics: Accuracy, Precision, Recall, F1-score

**Installation:**
1. Clone this repository:
   ```bash
   git clone https://github.com/jeyran-azizi/Deep-Learning.git
   cd Deep-Learning/cnn
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib scikit-learn
   ```

**Dataset:**
The dataset is automatically downloaded from Torchvision.
- Training: 80% of CIFAR-10 dataset
- Validation: 20% of CIFAR-10 dataset
- Test: Separate CIFAR-10 test set

**Model Architecture:**
The CNN consists of:
1. Conv2D (3→32) → ReLU → MaxPooling
2. Conv2D (32→64) → ReLU → MaxPooling
3. Fully Connected Layer (512 neurons) → ReLU → Dropout
4. Output Layer (10 classes)

**Usage:**
1️⃣ Train the Model
- Run the training script:
```bash
python train.py
```

2️⃣ Evaluate the Model
After training, test the model using:
```bash
python evaluate.py
```

3️⃣ Load Pretrained Model
To use the saved model:
```python
import torch
from models import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("cifar10_cnn.pth"))
model.eval()
```

**Results**
After training, the model achieved:
- Validation Accuracy: 74%

**Notes:**
- The trained model is saved as `cifar10_cnn.pth`.
- Modify `train.py` to tune hyperparameters.

**TODO:**
- Improve model architecture
- Add data augmentation techniques
- Experiment with different optimizers

**Acknowledgments:**
- PyTorch
- CIFAR-10 dataset (Krizhevsky, 2009)
