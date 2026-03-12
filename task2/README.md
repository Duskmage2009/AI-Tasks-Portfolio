NER + Animal Classification Pipeline

A two-model ML pipeline: NER extracts the animal name from text, CNN classifies the photo — then we compare them.

## How It Works

```
Input text: "There is a cow in the picture."
                ↓ NER model (DistilBERT)
             "cow"   ←→   [photo] → "cow"  (ResNet18 classifier)
                ↓
             True 
```



## Installation
```bash
pip install -r requirements.txt
```

## Step-by-Step Setup

### 1. Download the dataset
```bash
python download_dataset.py
```
Downloads Animals-10 (~28k photos, 10 classes) into `data/raw/`.  
If automatic download fails, download manually from [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10) and run:
```bash
python download_dataset.py --manual data/animals10/raw-img
```

### 2. Train the NER model
```bash
python ner_train.py
# Options:
python ner_train.py --epochs 5 --output_dir ner_model
```
Saves the model to `ner_model/`.

### 3. Train the image classifier
```bash
python classifier_train.py --data_dir data/raw
# Options:
python classifier_train.py --data_dir data/raw --epochs 10 --output_dir animal_classifier
```
Saves the model to `animal_classifier/`.

### 4. Run the pipeline
```bash
python pipeline.py --text "There is a cow in the picture" --image data/raw/cow/img_0001.jpg
# → ANSWER: True

python pipeline.py --text "I see a dog" --image data/raw/cat/img_0001.jpg
# → ANSWER: False
```

### 5. Run the demo notebook
```bash
jupyter notebook eda.ipynb
```

## Pipeline Demo (from demo notebook)

The pipeline is verified against 10 test cases covering:

| # | Test Case | Expected |
|---|-----------|----------|
| 1 | Standard match — text says "cow", photo has cow | `True` |
| 2 | Mismatch — text says "cat", photo has dog | `False` |
| 3 | Negation — "does not contain any spiders" + spider photo | `True` |
| 4 | Standard match — "butterfly" text + butterfly photo | `True` |
| 5 | Mismatch — "horse" text + chicken photo | `False` |
| 6 | Standard match — "elephant" + elephant photo | `True` |
| 7 | Synonym — "stallion" maps to "horse" | `True` |
| 8 | Standard match — "sheep" + sheep photo | `True` |
| 9 | Out-of-vocabulary — "bird" not in dataset | `False` |
| 10 | Negation + match — "no cat" + cat photo | `False` |

## Animal Classes
`dog` · `cat` · `horse` · `spider` · `butterfly` · `chicken` · `sheep` · `cow` · `squirrel` · `elephant`

## Models Used
- **NER**: `distilbert-base-uncased` fine-tuned for animal entity extraction
- **Classifier**: `ResNet18` with transfer learning, fine-tuned on Animals-10
