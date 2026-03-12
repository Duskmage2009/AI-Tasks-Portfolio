MNIST Image Classification

Three models for handwritten digit classification (MNIST dataset), wrapped in a single unified interface.

## Models
| Algorithm | Class | Accuracy |
|-----------|-------|----------|
| `rf`  | Random Forest | ~97% |
| `nn`  | Feed-Forward Neural Network | ~98% |
| `cnn` | Convolutional Neural Network | ~99% |



## Installation
```bash
pip install -r requirements.txt
```

## Running the Demo
```bash
jupyter notebook demo.ipynb
```
Open `demo.ipynb` and run all cells (`Shift+Enter` for each cell, or `Kernel → Restart & Run All`).

The demo will:
1. Load the MNIST dataset automatically
2. Train the CNN model on a small subset (2000 samples for speed)
3. Predict a random test image and visualize the result

## Usage Example
```python
from mnist_classifier import MnistClassifier

# Choose algorithm: "rf", "nn", or "cnn"
clf = MnistClassifier(algorithm="cnn")
clf.train(X_train, y_train)
predictions = clf.predict(X_test)
```

## How It Works
- `MnistClassifierInterface` — abstract base class  with `train()` and `predict()` methods
- Each model implements the interface independently
- `MnistClassifier` wraps all three models — the user only picks the algorithm name, everything else is the same
