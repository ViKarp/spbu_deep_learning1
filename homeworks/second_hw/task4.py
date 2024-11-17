import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


class FashionMNISTDatasetWithTransforms(torch.utils.data.Dataset):
    """
        A custom dataset wrapper that applies a series of transformations to the input images.
    """
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = ToPILImage()(image)
        if self.transforms:
            image = self.transforms(image)
        return image, label


class SimpleCNN(nn.Module):
    """
        A simple CNN model for classifying images from the FashionMNIST dataset.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Trains the CNN model on the training data and returns the training loss and accuracy for each epoch.
    """
    model.train()
    train_loss = []
    train_acc = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.unsqueeze(1)  # Добавляем канал
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_loss, train_acc


def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.unsqueeze(1)  # Добавляем канал
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    return test_loss, test_acc


def plot_metrics(train_loss, train_acc, test_loss, test_acc, title_suffix):
    epochs = len(train_loss)
    plt.figure(figsize=(12, 6))

    # Лосс
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss, label="Train Loss")
    plt.plot(range(1, epochs + 1), [test_loss] * epochs, label="Test Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Testing Loss {title_suffix}')
    plt.legend()

    # Точность
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_acc, label="Train Accuracy")
    plt.plot(range(1, epochs + 1), [test_acc] * epochs, label="Test Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training and Testing Accuracy {title_suffix}')
    plt.legend()

    plt.show()