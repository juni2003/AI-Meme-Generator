"""Dataset loading and preparation functions"""
import pandas as pd, numpy as np, spacy, sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from core.preprocessor import enhanced_preprocess_text

try:
    nlp = spacy.load('en_core_web_md')
except Exception as e:
    print(f"Install spacy: python -m spacy download en_core_web_md")
    sys.exit(1)

def load_dataset(csv_path):
    return pd.read_csv(csv_path)

def calculate_text_embeddings(df, status_callback=None):
    if status_callback:
        status_callback("Calculating embeddings...")
    
    embeddings = []
    for text in df['processed_text']:
        try:
            if text and len(str(text).strip()) > 0:
                doc = nlp(str(text))
                embeddings.append(doc.vector)
            else:
                embeddings.append(np.zeros(nlp.vocab.vectors.shape[1]))
        except:
            embeddings.append(np.zeros(nlp.vocab.vectors.shape[1]))
    
    return np.array(embeddings)

def extract_lda_topics(df, count_vectorizer, num_topics=10, status_callback=None):
    if status_callback:
        status_callback(f"Extracting {num_topics} topics...")
    
    try:
        doc_term_matrix = count_vectorizer.fit_transform(df['processed_text'])
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=10)
        lda.fit(doc_term_matrix)
        topic_distributions = lda.transform(doc_term_matrix)
        feature_names = count_vectorizer.get_feature_names_out()
        
        def get_topic_keywords(topic_idx, n_top_words=5):
            topic = lda.components_[topic_idx]
            top_word_indices = topic.argsort()[:-n_top_words-1:-1]
            return [feature_names[i] for i in top_word_indices]
        
        topic_keywords = {i: get_topic_keywords(i) for i in range(num_topics)}
        df['topic_distribution'] = list(topic_distributions)
        df['dominant_topic'] = df['topic_distribution'].apply(np.argmax)
        
        return df, lda, topic_keywords, True
    except Exception as e:
        if status_callback:
            status_callback(f"Topic modeling failed: {str(e)}")
        return df, None, None, False

def load_and_prepare_dataset(csv_path, status_callback=None):
    if status_callback:
        status_callback("Loading dataset...")
    
    df = load_dataset(csv_path)
    
    if status_callback:
        status_callback(f"Loaded {len(df)} rows.")
    
    # Combine text fields
    df['combined_text'] = df.apply(
        lambda row: ' '.join([str(text) for text in [
            row.get('text_ocr', ''), 
            row.get('text_corrected', '')
        ] if pd.notna(text)]), 
        axis=1
    )
    
    # Preprocess text
    if status_callback:
        status_callback("Preprocessing...")
    
    df['processed_text'] = df['combined_text'].apply(enhanced_preprocess_text)
    
    # TF-IDF vectors
    if status_callback:
        status_callback("Creating TF-IDF...")
    
    try:
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
        
        tfidf_arrays = []
        for i in range(tfidf_matrix.shape[0]):
            tfidf_arrays.append(tfidf_matrix[i].toarray()[0])
        
        df['tfidf_vector'] = tfidf_arrays
    except:
        df['tfidf_vector'] = [np.zeros(10) for _ in range(len(df))]
    
    # Count vectors for LDA
    if status_callback:
        status_callback("Creating Count vectors...")
    
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    
    # Calculate embeddings
    embeddings = calculate_text_embeddings(df, status_callback)
    
    # Calculate sentiment
    if status_callback:
        status_callback("Calculating sentiment...")
    
    if 'overall_sentiment' in df.columns:
        sentiment_map = {
            'very_negative': -1.0, 'negative': -0.5, 'neutral': 0.0, 
            'positive': 0.5, 'very_positive': 1.0
        }
        df['sentiment_score'] = df['overall_sentiment'].map(sentiment_map)
    else:
        df['sentiment_score'] = df['combined_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
    
    # Topic modeling
    try:
        df, lda_model, topic_keywords, has_topics = extract_lda_topics(
            df, count_vectorizer, num_topics=15, status_callback=status_callback
        )
    except Exception as e:
        if status_callback:
            status_callback(f"Topic modeling failed: {str(e)}")
        lda_model = None
        topic_keywords = None
        has_topics = False
    
    models = {
        'tfidf_vectorizer': tfidf_vectorizer,
        'count_vectorizer': count_vectorizer if has_topics else None,
        'lda_model': lda_model,
        'embeddings': embeddings,
        'topic_keywords': topic_keywords
    }
    
    if status_callback:
        status_callback("Complete!")
    
    return df, models