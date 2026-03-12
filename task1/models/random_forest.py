"""
random_forest.py — Random Forest classifier implementation using scikit-learn.

Random Forest is an ensemble of multiple decision trees. Each tree looks at 
different subsets of data, and the final prediction is based on a majority vote.
Pros: Fast training, no GPU required.
Cons: Lower accuracy on image data compared to neural networks (CNNs).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import MnistClassifierInterface


class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier based on scikit-learn's Random Forest.
    Input 28x28 images are flattened into a 784-feature vector.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Args:
            n_estimators (int): Number of trees in the forest (more trees = better accuracy, more memory).
            random_state (int): Seed for reproducibility.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1  # Use all available CPU cores for parallel processing
        )

    def _flatten(self, x: np.ndarray) -> np.ndarray:
        """
        Random Forest does not support 2D images — flattens to a 1D vector.
        (N, 28, 28) → (N, 784)
        """
        return x.reshape(x.shape[0], -1)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the Random Forest model on the training set."""
        print("Training Random Forest...")
        x_flat = self._flatten(x_train)
        self.model.fit(x_flat, y_train)
        print("Random Forest Training Complete!")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts digits for the given input images."""
        x_flat = self._flatten(x)
        return self.model.predict(x_flat)