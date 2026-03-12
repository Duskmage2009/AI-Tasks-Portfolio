NER + Animal Classification Pipeline
A two-model ML pipeline: NER extracts the animal name from text, CNN classifies the photo — then we compare them.
How It Works
Input text: "There is a cow in the picture."
                ↓ NER model (DistilBERT)
             "cow"   ←→   [photo] → "cow"  (ResNet18 classifier)
                ↓
             True ✅

Installation
bashpip install -r requirements.txt
Step-by-Step Setup
1. Download the dataset
bashpython download_dataset.py
Downloads Animals-10 (~28k photos, 10 classes) into data/raw/.
If automatic download fails, download manually from Kaggle and run:
bashpython download_dataset.py --manual data/animals10/raw-img
2. Train the NER model
bashpython ner_train.py
# Options:
python ner_train.py --epochs 5 --output_dir ner_model
Saves the model to ner_model/.
3. Train the image classifier
bashpython classifier_train.py --data_dir data/raw
# Options:
python classifier_train.py --data_dir data/raw --epochs 10 --output_dir animal_classifier
Saves the model to animal_classifier/.
4. Run the pipeline
bashpython pipeline.py --text "There is a cow in the picture" --image data/raw/cow/img_0001.jpg
# → ANSWER: True

python pipeline.py --text "I see a dog" --image data/raw/cat/img_0001.jpg
# → ANSWER: False
5. Run the demo notebook
bashjupyter notebook eda.ipynb
Pipeline Demo (from demo notebook)
The pipeline is verified against 10 test cases covering:
#Test CaseExpected1Standard match — text says "cow", photo has cowTrue2Mismatch — text says "cat", photo has dogFalse3Negation — "does not contain any spiders" + spider photoTrue4Standard match — "butterfly" text + butterfly photoTrue5Mismatch — "horse" text + chicken photoFalse6Standard match — "elephant" + elephant photoTrue7Synonym — "stallion" maps to "horse"True8Standard match — "sheep" + sheep photoTrue9Out-of-vocabulary — "bird" not in datasetFalse10Negation + match — "no cat" + cat photoFalse
Animal Classes
dog · cat · horse · spider · butterfly · chicken · sheep · cow · squirrel · elephant
Models Used

NER: distilbert-base-uncased fine-tuned for animal entity extraction
Classifier: ResNet18 with transfer learning, fine-tuned on Animals-10