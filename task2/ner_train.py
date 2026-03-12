"""
ner_train.py — NER Model Training for animal entity extraction.

What is NER (Named Entity Recognition)?
  It is a Natural Language Processing task of identifying named entities in text.
  Example: "There is a cow in the picture" -> identifies "cow" as an ANIMAL entity.

Model: distilbert-base-uncased (lightweight transformer, fast to train).
Approach: Token Classification — assigning B-ANIMAL, I-ANIMAL, or O labels to each token.

Usage:
  python ner_train.py
  python ner_train.py --output_dir my_ner_model --epochs 5
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------
# 1. Configuration
# ---------------------------------------------
DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_OUTPUT = "ner_model"
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-5
DEFAULT_BATCH = 16

# BIO-tagging scheme labels
LABEL2ID = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Dataset animal classes + synonyms for robust training
ANIMALS = [
    "dog", "dogs", "puppy", "puppies",
    "cat", "cats", "kitten", "kittens",
    "horse", "horses", "foal",
    "spider", "spiders",
    "butterfly", "butterflies",
    "chicken", "chickens", "hen", "hens", "rooster",
    "sheep", "lamb", "lambs",
    "cow", "cows", "bull", "calf",
    "squirrel", "squirrels",
    "elephant", "elephants",
]


# ---------------------------------------------
# 2. Synthetic Data Generation
# ---------------------------------------------

def generate_training_data(n_samples: int = 2000) -> list:
    """
    Generates synthetic sentences for NER training.
    Uses templates to expose the model to various ways of mentioning animals.
    """
    templates = [
        "There is a {animal} in the picture.",
        "I can see a {animal} in this image.",
        "The photo shows a {animal}.",
        "This is a picture of a {animal}.",
        "Look, there is a {animal} here.",
        "A {animal} is visible in the image.",
        "The image contains a {animal}.",
        "That looks like a {animal} to me.",
        "I think this is a {animal}.",
        "This photo has a {animal} in it.",
        "Can you see the {animal} in this picture?",
        "The {animal} is clearly visible here.",
        "Is that a {animal} in the photo?",
        "What a beautiful {animal}!",
        "There appears to be a {animal} here.",
        "I believe this image shows a {animal}.",
        "This is definitely a {animal}.",
        "That is a {animal} in the image.",
        "There is no {animal} in this picture.",
        "I see an animal but it is not a {animal}.",
    ]

    negative_templates = [
        "The weather looks nice today.",
        "I love this beautiful landscape.",
        "What an amazing view!",
        "This is a great photo.",
        "The colors in this image are stunning.",
    ]

    data = []
    for _ in range(n_samples):
        animal = random.choice(ANIMALS)

        # 80% animal-related sentences, 20% negative samples
        if random.random() < 0.8:
            template = random.choice(templates)
            sentence = template.format(animal=animal)
        else:
            sentence = random.choice(negative_templates)
            animal = None

        # Basic word-level tokenization
        tokens = sentence.replace(".", " .").replace(
            "?", " ?").replace("!", " !").split()
        labels = []
        for token in tokens:
            token_lower = token.lower().strip(".,!?")
            if animal and token_lower == animal:
                labels.append("B-ANIMAL")
            else:
                labels.append("O")

        data.append({"tokens": tokens, "labels": labels})

    return data


# ---------------------------------------------
# 3. PyTorch Dataset
# ---------------------------------------------

class NERDataset(Dataset):
    """NER Dataset — converts tokens and labels into tensors for the Transformer model."""

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Align labels with sub-tokens (DistilBert uses WordPiece tokenization)
        word_ids = encoding.word_ids()
        label_ids = []
        prev_word = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Special tokens — ignored in loss calculation
            elif word_id != prev_word:
                label_ids.append(LABEL2ID[labels[word_id]])  # First sub-token of a word
            else:
                # Subsequent sub-tokens — ignored to avoid label bias
                label_ids.append(-100)
            prev_word = word_id

        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(label_ids, dtype=torch.long),
        }


# ---------------------------------------------
# 4. Training Loop
# ---------------------------------------------

def train(output_dir=DEFAULT_OUTPUT, model_name=DEFAULT_MODEL,
          epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch_size=DEFAULT_BATCH, n_samples=2000):

    print(f"Base Model: {model_name}")
    print(f"Epochs: {epochs}, Learning Rate: {lr}, Batch Size: {batch_size}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Data preparation
    print("Generating synthetic training data...")
    all_data = generate_training_data(n_samples)
    split = int(len(all_data) * 0.9)
    train_data, val_data = all_data[:split], all_data[split:]
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}\n")

    # Model and Tokenizer loading
    print("Loading pre-trained model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    train_loader = DataLoader(NERDataset(
        train_data, tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NERDataset(
        val_data, tokenizer),   batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, total_steps // 10, total_steps)

    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += outputs.loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  Best model saved to -> '{output_dir}'")

    # Save label configuration for inference
    with open(os.path.join(output_dir, "label_config.json"), "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL,
                  "animals": ANIMALS}, f, indent=2)

    print(f"\nTraining complete! Model stored in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a NER model for animal entity extraction.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--n_samples",  type=int,   default=2000)
    args = parser.parse_args()

    train(args.output_dir, args.model_name, args.epochs,
          args.lr, args.batch_size, args.n_samples)