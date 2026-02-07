import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from preprocessing import clean_text, preprocess_text

df = df = pd.read_csv("data/Truth_Seeker_Model_Dataset.csv")

df['cleaned_tweet'] = df['tweet'].apply(clean_text)
df['processed_tweet'] = df['cleaned_tweet'].apply(preprocess_text)

X_text = df['processed_tweet']
y = df['BinaryNumTarget']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.3, random_state=42, stratify=y
)

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5, max_df=0.9)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

ann_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_svd.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

ann_model.fit(X_train_svd, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=[early_stop])

os.makedirs("models", exist_ok=True)
ann_model.save("models/final_ann_model.keras")
joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(svd, "models/svd_model.pkl")