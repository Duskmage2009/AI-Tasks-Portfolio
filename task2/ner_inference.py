"""
ner_inference.py — NER Model Inference: extracting animal entities from text.

Usage:
    python ner_inference.py --text "There is a cow in the picture"
    python ner_inference.py --text "I see a dog" --model_dir ner_model
"""

import argparse
import json
import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class NERInference:
    """
    A class to extract animal names from arbitrary English text using a trained NER model.

    Example:
        ner = NERInference("ner_model")
        animals = ner.extract_animals("There is a cow in the picture")
        # → ["cow"]
    """

    def __init__(self, model_dir: str = "ner_model"):
        """
        Args:
            model_dir: Directory containing the saved model (output from ner_train.py).
        """
        print(f"Loading NER model from '{model_dir}'...")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir).to(self.device)
        self.model.eval()

        # Load label configuration mapping
        config_path = os.path.join(model_dir, "label_config.json")
        with open(config_path) as f:
            config = json.load(f)
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        print("NER model loaded successfully!\n")

    def extract_animals(self, text: str) -> List[str]:
        """
        Extracts animal names from the input text.

        Args:
            text: Arbitrary English text.

        Returns:
            A list of identified animals (lowercase, unique).
            Returns an empty list if no animals are found.
        """
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert logits to label indices
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze().tolist())

        # Logic to reconstruct words from tokens tagged as B-ANIMAL or I-ANIMAL
        animals = []
        current_entity = []

        for token, pred_id in zip(tokens, predictions):
            label = self.id2label.get(pred_id, "O")

            # Ignore special tokens like [CLS], [SEP], [PAD]
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                if current_entity:
                    animals.append(" ".join(current_entity))
                    current_entity = []
                continue

            # Handle sub-word tokens starting with "##" (merge with previous token)
            if token.startswith("##"):
                if current_entity:
                    current_entity[-1] += token[2:]
                continue

            if label == "B-ANIMAL":
                if current_entity:
                    animals.append(" ".join(current_entity))
                current_entity = [token]
            elif label == "I-ANIMAL" and current_entity:
                current_entity.append(token)
            else:
                if current_entity:
                    animals.append(" ".join(current_entity))
                    current_entity = []

        if current_entity:
            animals.append(" ".join(current_entity))

        # Return unique lowercase results
        return list(set(a.lower() for a in animals))


def main():
    parser = argparse.ArgumentParser(
        description="Animal Entity Extraction (NER Inference)")
    parser.add_argument("--text",      required=True, help="Input text to analyze")
    parser.add_argument("--model_dir", default="ner_model",
                        help="Path to the trained NER model directory")
    args = parser.parse_args()

    ner = NERInference(args.model_dir)
    animals = ner.extract_animals(args.text)

    print(f"Input Text: {args.text}")
    print(f"Detected:   {animals if animals else 'None found'}")


if __name__ == "__main__":
    main()