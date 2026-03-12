"""
cnn.py — Convolutional Neural Network (CNN) implementation using PyTorch.

CNNs are designed to process pixel data by using convolutional filters to 
identify spatial patterns such as edges, corners, and shapes. 
This is typically the most accurate model for image classification tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import MnistClassifierInterface


class CNNModel(nn.Module):
    """
    CNN Architecture:
    Convolutional Layers (Feature Extraction) → Fully Connected Layers (Classification).
    """

    def __init__(self):
        super(CNNModel, self).__init__()

        # Feature extraction: extracting patterns from images
        self.conv_layers = nn.Sequential(
            # First convolutional block: 1 channel (grayscale) → 32 filters
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Spatial reduction: 28x28 → 14x14

            # Second convolutional block: 32 → 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Spatial reduction: 14x14 → 7x7
        )

        # Classification part: decision making based on extracted features
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                # 64 * 7 * 7 = 3136 features
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),             # Regularization to prevent overfitting
            nn.Linear(128, 10)           # 10 output classes (digits 0-9)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class CNNMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier wrapper for the PyTorch-based CNN model.
    """

    def __init__(self, epochs: int = 10, batch_size: int = 64, lr: float = 0.001):
        """
        Args:
            epochs (int): Number of training iterations.
            batch_size (int): Size of data batches.
            lr (float): Learning rate for the optimizer.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNModel().to(self.device)

    def _prepare_data(self, x: np.ndarray, y: np.ndarray = None):
        """
        Prepares data for the CNN.
        Reshapes to (N, 1, 28, 28) and normalizes pixels to [0, 1].
        """
        # Reshape: (N, 28, 28) or (N, 784) → (N, 1, 28, 28)
        x_reshaped = x.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        x_tensor = torch.FloatTensor(x_reshaped)

        if y is not None:
            y_tensor = torch.LongTensor(y.astype(np.int64))
            return TensorDataset(x_tensor, y_tensor)
        return x_tensor

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the CNN model."""
        print(f"Training CNN on {self.device}...")

        dataset = self._prepare_data(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0

            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            avg_loss = total_loss / len(loader)
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        print("CNN Training Complete!")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts digits for the given input array."""
        self.model.eval()
        x_tensor = self._prepare_data(x).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_tensor)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()