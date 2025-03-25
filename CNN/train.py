import torch
from torch import optim, nn
from models import SimpleCNN
from dataset import train_loader, validation_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = SimpleCNN().to(device)

# Hyperparameters
learning_rate = 0.001
num_epochs = 20

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_epoch_loss = val_loss / len(validation_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(validation_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, '
              f'Training Loss: {epoch_loss:.4f}, '
              f'Validation Loss: {val_epoch_loss:.4f}, '
              f'Validation Accuracy: {val_epoch_acc:.4f}')

# Train and save the model
train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), 'cifar10_model.pth')
