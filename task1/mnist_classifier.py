"""
mnist_classifier.py — Main Wrapper Class: MnistClassifier.

This is a single entry point for all three models.
The user simply calls MnistClassifier(algorithm="cnn") and interacts with it — 
unaware of the specific implementation details inside. 
This follows the "Facade" design pattern.
"""

import numpy as np
from models import (
    RandomForestMnistClassifier,
    NeuralNetworkMnistClassifier,
    CNNMnistClassifier,
    MnistClassifierInterface
)


class MnistClassifier:
    """
    Unified interface for all MNIST classifiers.

    Usage:
        clf = MnistClassifier(algorithm="cnn")
        clf.train(X_train, y_train)
        predictions = clf.predict(X_test)

    Available algorithms:
        "rf"  — Random Forest
        "nn"  — Feed-Forward Neural Network
        "cnn" — Convolutional Neural Network
    """

    # Dictionary mapping string keys to model classes
    ALGORITHMS = {
        "rf":  RandomForestMnistClassifier,
        "nn":  NeuralNetworkMnistClassifier,
        "cnn": CNNMnistClassifier,
    }

    def __init__(self, algorithm: str, **kwargs):
        """
        Args:
            algorithm: one of "rf", "nn", "cnn"
            **kwargs: additional parameters passed directly to the chosen model
                      e.g.: MnistClassifier("cnn", epochs=5)
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: '{algorithm}'. "
                f"Available options: {list(self.ALGORITHMS.keys())}"
            )

        self.algorithm = algorithm
        # Instantiate the requested model — all follow the same interface
        self.model: MnistClassifierInterface = self.ALGORITHMS[algorithm](**kwargs)
        print(f"Classifier initialized: {algorithm.upper()}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X_train: images 28x28, shape (N, 28, 28), values 0-255
            y_train: digit labels 0-9, shape (N,)
        """
        self.model.train(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict digits for given images.

        Args:
            X: images 28x28, shape (N, 28, 28)

        Returns:
            np.ndarray: predicted digits, shape (N,)
        """
        return self.model.predict(X)