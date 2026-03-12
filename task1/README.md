MNIST Image Classification
Three models for handwritten digit classification (MNIST dataset), wrapped in a single unified interface.
Models
AlgorithmClassAccuracyrfRandom Forest~97%nnFeed-Forward Neural Network~98%cnnConvolutional Neural Network~99%

Installation
bashpip install -r requirements.txt
Running the Demo
bashjupyter notebook demo.ipynb
Open demo.ipynb and run all cells (Shift+Enter for each cell, or Kernel → Restart & Run All).
The demo will:

Load the MNIST dataset automatically
Train the CNN model on a small subset (2000 samples for speed)
Predict a random test image and visualize the result

Usage Example
pythonfrom mnist_classifier import MnistClassifier

# Choose algorithm: "rf", "nn", or "cnn"
clf = MnistClassifier(algorithm="cnn")
clf.train(X_train, y_train)
predictions = clf.predict(X_test)
How It Works

MnistClassifierInterface — abstract base class (like a Java interface) with train() and predict() methods
Each model implements the interface independently
MnistClassifier wraps all three models — the user only picks the algorithm name, everything else is the same