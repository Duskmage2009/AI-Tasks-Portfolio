"""
pipeline.py — Main Pipeline: Text + Image → True/False.

Workflow:
  1. NER model identifies the animal in the text (e.g., "There is a cow" → "cow").
  2. Classifier identifies the animal in the image → "cow".
  3. Comparison: Do they match? → True / False.

Usage:
  python pipeline.py --text "There is a cow in the picture" --image cow.jpg
  python pipeline.py --text "I see a dog" --image photo.jpg --ner_dir ner_model --clf_dir animal_classifier
"""

import argparse
import os
from typing import List, Dict, Any
from ner_inference import NERInference
from classifier_inference import AnimalClassifierInference


# Synonym Dictionary: maps various terms to the standard dataset class names
SYNONYMS = {
    # dog
    "dog": "dog", "dogs": "dog", "puppy": "dog", "puppies": "dog", "canine": "dog",
    # cat
    "cat": "cat", "cats": "cat", "kitten": "cat", "kittens": "cat", "feline": "cat",
    # horse
    "horse": "horse", "horses": "horse", "foal": "horse", "pony": "horse", "mare": "horse", "stallion": "horse",
    # spider
    "spider": "spider", "spiders": "spider", "arachnid": "spider",
    # butterfly
    "butterfly": "butterfly", "butterflies": "butterfly",
    # chicken
    "chicken": "chicken", "chickens": "chicken", "hen": "chicken", "hens": "chicken",
    "rooster": "chicken", "chick": "chicken",
    # sheep
    "sheep": "sheep", "lamb": "sheep", "lambs": "sheep", "ewe": "sheep",
    # cow
    "cow": "cow", "cows": "cow", "bull": "cow", "calf": "cow", "cattle": "cow", "ox": "cow",
    # squirrel
    "squirrel": "squirrel", "squirrels": "squirrel",
    # elephant
    "elephant": "elephant", "elephants": "elephant",
}


class AnimalPipeline:
    """
    Pipeline: accepts text + image path, returns a boolean verification result.

    True  — The animal mentioned in the text matches the animal in the image.
    False — No match found, or no animal detected in the text.
    """

    def __init__(self, ner_dir: str = "ner_model", clf_dir: str = "animal_classifier"):
        """
        Args:
            ner_dir: Directory containing the trained NER model.
            clf_dir: Directory containing the trained image classifier.
        """
        print("Initializing Pipeline...\n")
        self.ner = NERInference(ner_dir)
        self.clf = AnimalClassifierInference(clf_dir)
        print("Pipeline is ready!\n")

    def _normalize(self, animal: str) -> str:
        """Normalizes animal names using the SYNONYMS dictionary."""
        clean_name = animal.lower().strip()
        return SYNONYMS.get(clean_name, clean_name)

    def predict(self, text: str, image_path: str, verbose: bool = True) -> bool:
        """
        Main pipeline execution method.

        Args:
            text: User input text (e.g., "There is a cow in the picture").
            image_path: Path to the input image file.
            verbose: Whether to print execution details (for debugging).

        Returns:
            True if the text-extracted animal matches the image-detected animal, False otherwise.
        """
        # Step 1: Extract animals from text
        animals_in_text = self.ner.extract_animals(text)
        normalized_text = [self._normalize(a) for a in animals_in_text]

        if verbose:
            print(f"Input Text:       '{text}'")
            print(f"NER Entities:     {animals_in_text} → Normalized: {normalized_text}")

        # If no animal is mentioned in the text, return False
        if not normalized_text:
            if verbose:
                print("Result: False (No animal entities detected in text)\n")
            return False

        # Step 2: Classify the image
        clf_result = self.clf.predict(image_path)
        animal_image = clf_result["class"]
        confidence = clf_result["confidence"]

        if verbose:
            print(f"Image Prediction: {animal_image} (Confidence: {confidence*100:.1f}%)")
            print(f"Top-3 Classes:    {clf_result['top3']}")

        # Step 3: Comparison logic
        # Check for negation words to handle cases like "There is no dog"
        negation_words = ["no", "not", "without", "isn't", "aren't", "doesn't"]
        has_negation = any(word in text.lower().split() for word in negation_words)

        match = animal_image in normalized_text
        # Logic: (match exists and no negation) OR (no match exists but there is negation)
        result = (match and not has_negation) or (not match and has_negation)

        if verbose:
            print(f"\nMatch Found:     {match}")
            print(f"Negation Detected: {has_negation}")
            print(f"Final Decision:   {'True' if result else 'False'}\n")

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Animal Verification Pipeline: Cross-referencing text entities with image content."
    )
    parser.add_argument("--text",    required=True, help="User text input")
    parser.add_argument("--image",   required=True, help="Path to the image file")
    parser.add_argument("--ner_dir", default="ner_model",        help="Directory of the NER model")
    parser.add_argument("--clf_dir", default="animal_classifier", help="Directory of the image classifier")
    parser.add_argument("--quiet",   action="store_true",         help="Suppress detailed logs, output result only")
    args = parser.parse_args()

    pipeline = AnimalPipeline(args.ner_dir, args.clf_dir)
    result = pipeline.predict(args.text, args.image, verbose=not args.quiet)

    print(f"FINAL RESULT: {result}")


if __name__ == "__main__":
    main()