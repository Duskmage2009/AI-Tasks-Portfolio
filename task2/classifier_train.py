"""
classifier_train.py — Animal Image Classifier Training.

Approach: Transfer Learning with ResNet18.
  We leverage a model pre-trained on ImageNet (1.2M images) and 
  fine-tune the final layer for our 10 animal classes.
  This is significantly faster and more accurate than training from scratch.

Usage:
  python classifier_train.py --data_dir data/raw
  python classifier_train.py --data_dir data/raw --epochs 10 --output_dir my_model
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ---------------------------------------------
# Configuration
# ---------------------------------------------
DEFAULT_DATA = "data/raw"
DEFAULT_OUTPUT = "animal_classifier"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
IMG_SIZE = 224  # ResNet standard input size


# ---------------------------------------------
# Data Transformations
# ---------------------------------------------

def get_transforms():
    """
    Data Augmentation — artificially increasing dataset diversity 
    through random cropping, flipping, and color adjustments.
    This helps prevent overfitting and improves generalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        # Normalize using standard ImageNet mean and standard deviation
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


# ---------------------------------------------
# Model Architecture
# ---------------------------------------------

def build_model(num_classes: int):
    """
    ResNet18 with frozen backbone and a custom classification head.

    Transfer Learning: freeze feature extraction layers (trained on ImageNet)
    and only train the final linear layers for our specific classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all parameters in the backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected (fc) layer
    # Originally 1000 ImageNet classes -> mapping to our num_classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    return model


# ---------------------------------------------
# Training Logic
# ---------------------------------------------

def train(data_dir=DEFAULT_DATA, output_dir=DEFAULT_OUTPUT,
          epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH, lr=DEFAULT_LR):

    print(f"Dataset path: {data_dir}")
    print(f"Params: Epochs={epochs}, LR={lr}, Batch={batch_size}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset from folder (Expected structure: data/raw/<class_name>/)
    train_tf, val_tf = get_transforms()
    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)

    # Extract class names from directory structure
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}\n")

    # Split: 80% Training / 20% Validation
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Override validation transform (exclude augmentations)
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Model, Loss Function, and Optimizer
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Train only the parameters of the new head (fc)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs, dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} — "
              f"Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save the checkpoint with best validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(
                output_dir, "best_model.pth"))
            print(f"  Best model saved (Val Acc: {val_acc:.2f}%)")

    # Store class mapping for inference
    config = {"classes": classes,
              "num_classes": num_classes, "img_size": IMG_SIZE}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining Complete! Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an animal image classifier using ResNet18 Transfer Learning.")
    parser.add_argument("--data_dir",   default=DEFAULT_DATA)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    train(args.data_dir, args.output_dir,
          args.epochs, args.batch_size, args.lr)