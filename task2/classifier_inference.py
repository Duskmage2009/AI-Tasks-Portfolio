"""
classifier_inference.py — Animal Classifier Inference: identifying animals in images.

Usage:
    python classifier_inference.py --image path/to/photo.jpg
    python classifier_inference.py --image photo.jpg --model_dir animal_classifier
"""

import argparse
import json
import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class AnimalClassifierInference:
    """
    Class for performing image classification to identify animals.

    Example:
        clf = AnimalClassifierInference("animal_classifier")
        result = clf.predict("cow.jpg")
        # → {"class": "cow", "confidence": 0.95, "top3": [("cow", 0.95), ...]}
    """

    def __init__(self, model_dir: str = "animal_classifier"):
        """
        Args:
            model_dir: Directory containing the saved model and config (output from classifier_train.py).
        """
        print(f"Loading classifier from '{model_dir}'...")

        # Load configuration file
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        self.classes = config["classes"]
        self.num_classes = config["num_classes"]
        self.img_size = config["img_size"]

        # Reconstruct the architecture used during training
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

        # Load trained weights
        weights_path = os.path.join(model_dir, "best_model.pth")
        self.model.load_state_dict(torch.load(
            weights_path, map_location=self.device))
        self.model.eval()

        # Inference-time transforms (Normalization only, no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        print(f"Classifier loaded successfully! Classes: {self.classes}\n")

    def _build_model(self) -> nn.Module:
        """Recreates the ResNet18 architecture with the custom classification head."""
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes),
        )
        return model.to(self.device)

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predicts the animal class for a given image.

        Args:
            image_path: Path to the image file (jpg, png, etc.)

        Returns:
            A dictionary containing:
            {
                "class": "cow",           # Predicted class name
                "confidence": 0.95,       # Model confidence score (0-1)
                "top3": [                 # Top 3 predictions with scores
                    ("cow", 0.95),
                    ("sheep", 0.03),
                    ("horse", 0.01)
                ]
            }
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(tensor)
            # Apply Softmax to get class probabilities
            probs = torch.softmax(outputs, dim=1).squeeze()

        # Extract Top-3 predictions
        top3_probs, top3_idx = torch.topk(probs, min(3, self.num_classes))
        top3 = [(self.classes[i], float(p))
                for i, p in zip(top3_idx, top3_probs)]

        return {
            "class":      top3[0][0],
            "confidence": top3[0][1],
            "top3":       top3,
        }

    def predict_class_only(self, image_path: str) -> str:
        """Fast method returning only the predicted class name."""
        return self.predict(image_path)["class"]


def main():
    parser = argparse.ArgumentParser(
        description="Animal Image Classification Inference")
    parser.add_argument("--image",     required=True, help="Path to input image")
    parser.add_argument(
        "--model_dir", default="animal_classifier", help="Directory of the trained model")
    args = parser.parse_args()

    clf = AnimalClassifierInference(args.model_dir)
    result = clf.predict(args.image)

    print(f"Image:      {args.image}")
    print(f"Prediction: {result['class']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Top-3:      {result['top3']}")


if __name__ == "__main__":
    main()