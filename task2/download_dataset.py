import opendatasets as od
import os
import shutil
from pathlib import Path

def download_animals10():
    """
    Downloads the Animals10 dataset from Kaggle, renames Italian labels to English,
    and organizes the directory structure for training.
    """
    # Create target directory structure
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle dataset URL
    dataset_url = "https://www.kaggle.com/datasets/alessiocorrado99/animals10"

    print("--- Starting download from Kaggle ---")
    # Note: Requires Kaggle Username and API Key (kaggle.json)
    od.download(dataset_url, data_dir)

    # Dataset default path: data/animals10/raw-img
    source_path = data_dir / "animals10" / "raw-img"

    # Label mapping: Italian folder names -> English class names
    label_map = {
        "cane": "dog", "cavallo": "horse", "elefante": "elephant",
        "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat",
        "mucca": "cow", "pecora": "sheep", "ragno": "spider", "scoiattolo": "squirrel"
    }

    print("--- Renaming classes and moving files... ---")
    for it_name, en_name in label_map.items():
        src = source_path / it_name
        dst = raw_dir / en_name
        
        if src.exists():
            # If destination already exists, remove it to ensure a clean copy
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(src, dst)
            print(f"Class '{en_name}' processed.")

    # Cleanup: Remove original download folder
    shutil.rmtree(data_dir / "animals10")
    print(f"\n✅ SUCCESS! All 10 classes are organized in {raw_dir.absolute()}")


if __name__ == "__main__":
    download_animals10()