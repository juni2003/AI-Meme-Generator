# AI Meme Generator - Similarity Evaluation Only

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Custom project modules (make sure these paths and functions are correctly defined in your project)
from core.preprocessor import enhanced_preprocess_text
from utils.data_loader import load_dataset

# Paths
DATASET_PATH = 'data/labels.csv'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_data(csv_path):
    """
    Loads the dataset and combines + cleans multiple text fields into a single processed text column.
    """
    df = load_dataset(csv_path)
    df['combined_text'] = df.apply(
        lambda row: ' '.join([str(text) for text in [
            row.get('text_ocr', ''), row.get('text_corrected', '')
        ] if pd.notna(text)]),
        axis=1
    )
    df['processed_text'] = df['combined_text'].apply(enhanced_preprocess_text)
    return df

def evaluate_similarity(X_test, y_test, df, tfidf_vectorizer, k=5):
    """
    Evaluates how often the true meme image is found in the top-k most similar captions based on cosine similarity.
    """
    all_templates_tfidf = tfidf_vectorizer.transform(df['processed_text'])
    correct = 0
    for test_text, true_template in zip(X_test, y_test):
        test_tfidf = tfidf_vectorizer.transform([test_text])
        similarities = cosine_similarity(test_tfidf, all_templates_tfidf).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        top_templates = df.iloc[top_indices]['image_name'].values
        if true_template in top_templates:
            correct += 1
    return (correct / len(X_test)) * 100

def main():
    print("="*50)
    print("MEME GENERATOR - SIMILARITY MODEL EVALUATION")
    print("="*50)

    # Step 1: Load and preprocess data
    df = preprocess_data(DATASET_PATH)

    # Step 2: Prepare TF-IDF vectorizer using training data
    X = df['processed_text']
    y = df['image_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf.fit(X_train)

    # Step 3: Evaluate the similarity matching model
    print("Evaluating similarity-based model...")
    
    score = evaluate_similarity(X_test, y_test, df, tfidf)
    print(f"Top 5 similarity accuracy: {score:.2f}%")

    # Step 4: Save TF-IDF model for future use
    with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(tfidf, f)
    print("TF-IDF vectorizer saved to disk.")

if __name__ == "__main__":
    main()