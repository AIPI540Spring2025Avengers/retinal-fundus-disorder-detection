'''
main script/notebook to run project / user interface
'''
'''
main script/notebook to run project / user interface
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
#mport seaborn as sns
import os

def main():
    print("Starting the script...")
    # Set device
    print("CUDA Available:", torch.cuda.is_available())
    print("MPS Available:", torch.backends.mps.is_available())  # For Mac M1/M2 users
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Current Device:", device)

    # Image size and paths
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    train_path = '/Users/aadi/Desktop/Deep Learning/Computer Vision/Retinal Fundus Images/train'
    test_path = '/Users/aadi/Desktop/Deep Learning/Computer Vision/Retinal Fundus Images/test'
    val_path = '/Users/aadi/Desktop/Deep Learning/Computer Vision/Retinal Fundus Images/val'

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load pretrained EfficientNetB4
    base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    base_model.classifier = nn.Sequential(
        nn.BatchNorm1d(1792),
        nn.Linear(1792, 256),
        nn.ReLU(),
        nn.Dropout(0.45),
        nn.Linear(256, 11)  # Output classes = 11
    )
    model = base_model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.001)

    # Train the model
    train_losses, val_losses, train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

    # Save the model
    torch.save(model.state_dict(), "efficientnet_b4_retinal.pth")
    print("Model saved as efficientnet_b4_retinal.pth")

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_losses, val_losses = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc.append(correct / total)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc.append(val_correct / val_total)

        print("Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}".format(epoch+1, epochs, train_loss, train_acc[-1], val_loss, val_acc[-1]))

    return train_losses, val_losses, train_acc, val_acc

if __name__ == "__main__":
    main()
