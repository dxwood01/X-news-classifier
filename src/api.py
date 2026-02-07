from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import tensorflow as tf
from src.preprocessing import clean_text, preprocess_text

tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
svd = joblib.load("models/svd_model.pkl")
model = tf.keras.models.load_model("models/final_ann_model.keras")

app = FastAPI(title="X News Classifier")

class Tweet(BaseModel):
    text: str

@app.post("/predict")
def predict_fake_news(tweet: Tweet):
    processed = preprocess_text(clean_text(tweet.text))
    tfidf_vec = tfidf_vectorizer.transform([processed])
    svd_vec = svd.transform(tfidf_vec)
    pred_prob = model.predict(svd_vec, verbose=0)
    
    probability = float(pred_prob[0][0])
    label = "Real" if probability > 0.5 else "Fake"
    
    confidence = abs(probability - 0.5) * 2
    
    if confidence < 0.3:
        confidence_level = "Low"
    elif confidence < 0.7:
        confidence_level = "Medium"
    else:
        confidence_level = "High"
    
    return {
        "prediction": label,
        "probability": probability,
        "confidence_level": confidence_level
    }