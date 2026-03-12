"""
neural_network.py — Feed-Forward Neural Network (MLP) implementation using PyTorch.

This is a classic Multi-Layer Perceptron (MLP): input layer → hidden layers → output layer.
Each neuron is connected to every neuron in the next layer (hence "fully connected").
Generally performs better than Random Forest on image data but is less efficient than CNN.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import MnistClassifierInterface


class FFNNModel(nn.Module):
    """
    Neural Network Architecture (Layer definitions).
    784 (input) → 256 → 128 → 10 (output, one for each digit).
    """

    def __init__(self):
        super(FFNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),   # First hidden layer
            nn.ReLU(),             # Activation function (rectified linear unit)
            nn.Dropout(0.2),       # Randomly deactivates 20% of neurons to prevent overfitting
            nn.Linear(256, 128),   # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)     # Output layer: 10 classes (digits 0-9)
        )

    def forward(self, x):
        """Forward pass: data flows through all layers."""
        return self.network(x)


class NeuralNetworkMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier wrapper for the PyTorch-based Feed-Forward Neural Network.
    """

    def __init__(self, epochs: int = 10, batch_size: int = 64, lr: float = 0.001):
        """
        Args:
            epochs (int): Number of passes through the entire dataset.
            batch_size (int): Number of images processed per step.
            lr (float): Learning rate — step size for weight updates.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        # Automatic hardware selection: GPU (CUDA) if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FFNNModel().to(self.device)

    def _prepare_data(self, x: np.ndarray, y: np.ndarray = None):
        """
        Converts NumPy arrays to PyTorch tensors and normalizes pixels.
        Reshapes input to flat vectors (N, 784).
        """
        # Flattening: (N, 28, 28) → (N, 784) and normalization 0-255 → 0.0-1.0
        x_flat = x.reshape(x.shape[0], -1).astype(np.float32) / 255.0
        x_tensor = torch.FloatTensor(x_flat)

        if y is not None:
            y_tensor = torch.LongTensor(y.astype(np.int64))
            return TensorDataset(x_tensor, y_tensor)
        return x_tensor

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the Feed-Forward Neural Network."""
        print(f"Training Neural Network on {self.device}...")

        dataset = self._prepare_data(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # CrossEntropyLoss: standard loss function for multi-class classification
        criterion = nn.CrossEntropyLoss()
        # Adam: adaptive optimizer that manages learning rate automatically
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()           # Reset gradients
                outputs = self.model(x_batch)   # Forward pass
                loss = criterion(outputs, y_batch)  # Calculate error
                loss.backward()                 # Backward pass (backpropagation)
                optimizer.step()                # Update weights

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_loss:.4f}")

        print("Neural Network Training Complete!")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts digits for the given input array."""
        self.model.eval()  # Switch to inference mode (disables Dropout)
        x_tensor = self._prepare_data(x).to(self.device)

        with torch.no_grad():  # Disable gradient tracking for faster inference
            outputs = self.model(x_tensor)
            # Pick the index of the highest probability
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()