import torch
from models import SimpleCNN
from dataset import test_loader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cifar10_model.pth'))
model.eval()

# Class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Compute classification report
report = classification_report(y_true, y_pred, target_names=classes, digits=4)
print(report)

# Function to print additional evaluation metrics
def evaluate_model(model, test_loader):
    model.eval()
    test_corrects = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for inputs, labels_batch in test_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels_batch.data)
            predictions.extend(preds.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())

    test_accuracy = test_corrects.double() / len(test_loader.dataset)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

evaluate_model(model, test_loader)
