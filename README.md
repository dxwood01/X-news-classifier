# X News Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

Automated fake news detection for X (Twitter) using NLP and Deep Learning. Achieves **97% accuracy** with a production-ready REST API.

---

## Overview

This project classifies tweets as **Real** or **Fake** using:

- Text preprocessing (cleaning, lemmatization)
- TF-IDF vectorization + SVD dimensionality reduction
- Artificial Neural Network (256→128→1 architecture)
- FastAPI for real-time predictions

**Performance:** 97.23% accuracy, 0.97 F1-score on 134K tweets

---

## Dataset

**Source:** [Truth Seeker Twitter Dataset 2023](https://www.kaggle.com/datasets/paulinepeps/truth-seeker-dataset-2023-truthseeker2023)

- 134K tweets (balanced)
- Download and place `Truth_Seeker_Model_Dataset.csv` in `data/` folder

---

## Pre-trained Models Included

This repository includes **pre-trained models** so you can use the API immediately without training:

- `models/final_ann_model.keras` - Trained neural network
- `models/tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer
- `models/svd_model.pkl` - Fitted SVD transformer

### Quick Start

```bash
# Clone and install
git clone https://github.com/dxwood01/X-news-classifier.git
cd X-news-classifier
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Run API immediately
uvicorn src.api:app --reload
```

Visit `http://localhost:8000/docs` and start predicting!

---

## Installation & Usage

### Option 1: Use Pre-trained Models (Recommended)

Follow the Quick Start above - models are already included!

### Option 2: Train from Scratch

If you want to retrain the models:

1. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/paulinepeps/truth-seeker-dataset-2023-truthseeker2023)
2. **Place** `Truth_Seeker_Model_Dataset.csv` in `data/` folder
3. **Train:**

```bash
   python src/train_model.py
```

This creates: `models/final_ann_model.keras`, `models/tfidf_vectorizer.pkl`, `models/svd_model.pkl`

---

## Making Predictions

### Via API (Interactive Docs)

1. Start the API: `uvicorn src.api:app --reload`
2. Open browser: `http://localhost:8000/docs`
3. Try the `/predict` endpoint

### Via cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news: Major discovery announced!"}'
```

**Response:**

```json
{
  "prediction": "Real",
  "probability": 0.923,
  "confidence_level": "High"
}
```

### Via Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Your tweet text here"}
)
print(response.json())
```

### Direct Python (Without API)

```python
from src.preprocessing import clean_text, preprocess_text
import joblib
import tensorflow as tf

# Load models
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
svd = joblib.load('models/svd_model.pkl')
model = tf.keras.models.load_model('models/final_ann_model.keras')

# Predict
tweet = "Your tweet text here"
processed = preprocess_text(clean_text(tweet))
vec = svd.transform(tfidf.transform([processed]))
pred = model.predict(vec, verbose=0)[0][0]
print("Real" if pred > 0.5 else "Fake", f"({pred:.2%})")
```

---

## Project Structure

```
X-news-classifier/
├── data/
│   └── .gitkeep                       # Dataset goes here (download separately)
├── models/                            # Pre-trained models
│   ├── final_ann_model.keras          # Trained ANN
│   ├── tfidf_vectorizer.pkl           # Fitted TF-IDF vectorizer
│   └── svd_model.pkl                  # Fitted SVD transformer
├── notebooks/
│   └── X_news_classifier.ipynb        # Full analysis & visualization
├── src/
│   ├── api.py                         # FastAPI endpoint
│   ├── preprocessing.py               # Text cleaning functions
│   └── train_model.py                 # Model training script
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Results

### Model Comparison

| Model                 | Test Accuracy | Test F1-Score |
| --------------------- | ------------- | ------------- |
| Logistic Regression   | 96.24%        | 0.9631        |
| Linear SVC            | 96.89%        | 0.9695        |
| Passive Aggressive    | 95.87%        | 0.9594        |
| SGD Classifier        | 96.15%        | 0.9621        |
| **ANN (Final Model)** | **97.23%**    | **0.9738**    |


### Example Predictions

| Tweet                                           | Prediction | Confidence | Notes              |
| ----------------------------------------------- | ---------- | ---------- | ------------------ |
| "CLICK HERE NOW!!! WIN $1000000!!!"             | Fake       | High       | Clear clickbait    |
| "The President announced new policy changes"    | Real       | Low        | Needs more context |
| "Research published in Nature journal shows..." | Fake       | High       | Short phrase       |

---

## Limitations

- **Context matters**: Short phrases may lack sufficient context for accurate classification
- **Dataset-specific**: Trained on 2023 Twitter data with specific linguistic patterns
- **Educational project**: Demonstrates ML skills; not production-ready without further validation
- **Confidence levels**: Use the `confidence_level` field to assess prediction reliability
- **Language**: English only

---

## Future Work

- [ ] Implement transformer models (BERT, RoBERTa)
- [ ] Add ensemble methods
- [ ] Deploy to cloud (AWS, GCP, Heroku)
- [ ] Create web interface
- [ ] Multilingual support
- [ ] Add explainability (LIME, SHAP)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Sulman Dawood**

- GitHub: [@dxwood01](https://github.com/dxwood01)
- LinkedIn: [Sulman Dawood](https://linkedin.com/in/sulman-dawood-552b5234a)

---

## Acknowledgments

- **Dataset**: Truth Seeker Twitter Dataset 2023 from [Kaggle](https://www.kaggle.com/datasets/paulinepeps/truth-seeker-dataset-2023-truthseeker2023)

---

**If you find this project useful, please consider giving it a star ⭐**

**Found a bug or have suggestions?** [Open an issue](https://github.com/dxwood01/X-news-classifier/issues)
