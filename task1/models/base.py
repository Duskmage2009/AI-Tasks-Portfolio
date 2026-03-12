"""
base.py — Base interface for all MNIST classifiers.

This module defines the Abstract Base Class (ABC) that ensures consistency 
across different machine learning models in the project.
"""

from abc import ABC, abstractmethod
import numpy as np


class MnistClassifierInterface(ABC):
    """
    Abstract Base Class (Interface) for all MNIST classifiers.
    
    Every model implementation MUST implement the train and predict methods 
    to ensure compatibility with the evaluation pipeline.
    """

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model using the provided dataset.

        Args:
            x_train (np.ndarray): Training images, shape (N, 28, 28) or (N, 784).
            y_train (np.ndarray): Target labels (digits 0-9), shape (N,).
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given input images.

        Args:
            x (np.ndarray): Input images, shape (N, 28, 28) or (N, 784).

        Returns:
            np.ndarray: An array of predicted digits, shape (N,).
        """
        pass