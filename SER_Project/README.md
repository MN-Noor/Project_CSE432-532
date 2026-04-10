# SER Project — Speech Emotion Recognition with MiniLearn

A complete Speech Emotion Recognition system built on the RAVDESS dataset,
featuring **MiniLearn** — a from-scratch mini scikit-learn library.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install MiniLearn in editable mode
pip install -e .

# 4. Download RAVDESS audio-only data
python download_data.py

# 5. Extract features
# e.g. python extract_features.py --data_dir data/ --output features.csv

# 6. Open the classification notebook
jupyter notebook notebooks/01_classification.ipynb
```

## Project Structure

```
SER_Project/
├── minilearn/                  # From-scratch ML library
│   ├── classifiers/            # LR, KNN, NB, Decision Tree
│   ├── preprocessing/          # StandardScaler, train_test_split
│   └── metrics/                # accuracy, precision, recall, F1, confusion matrix
├── notebooks/
│   └── 01_classification.ipynb # End-to-end SER classification demo
├── extract_features.py         # Audio → feature CSV pipeline
├── download_data.py            # Dataset download helper
├── requirements.txt            # Python dependencies (one example)
└── README.md
```

## Dataset

RAVDESS Audio-Only — 2,452 files (1,440 speech + 1,012 song) from 24 actors,
8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

Source: https://zenodo.org/records/1188976

## MiniLearn

Import it like scikit-learn:

```python
from minilearn.classifiers import LogisticRegression, KNN, GaussianNaiveBayes
from minilearn.preprocessing import StandardScaler, train_test_split
from minilearn.metrics import accuracy_score, f1_score, confusion_matrix
```
