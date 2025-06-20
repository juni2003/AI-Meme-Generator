import os, numpy as np, pickle
import spacy
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from core.preprocessor import enhanced_preprocess_text

try:
    nlp = spacy.load('en_core_web_md')
except:
    import sys
    print("Error: Install spacy model: python -m spacy download en_core_web_md")
    sys.exit(1)

MODELS_PATH = os.path.join('models', 'similarity_models.pkl')

def analyze_input(user_input):
    doc = nlp(user_input)
    
    keywords = []
    entities = []
    
    for ent in doc.ents:
        entities.append(ent.text.lower())
    
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop:
            keywords.append(token.lemma_.lower())
    
    blob = TextBlob(user_input)
    sentiment_score = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    important_words = [token.text.lower() for token in doc if not token.is_stop and len(token.text) > 2]
    lemmatized = " ".join([token.lemma_.lower() for token in doc if not token.is_stop])
    
    return {
        'keywords': ' '.join(keywords),
        'entities': entities,
        'important_words': important_words,
        'raw_input': user_input,
        'sentiment': sentiment_score,
        'subjectivity': subjectivity,
        'processed_text': enhanced_preprocess_text(user_input),
        'lemmatized': lemmatized,
        'vector': doc.vector
    }

def load_similarity_models():
    if os.path.exists(MODELS_PATH):
        try:
            with open(MODELS_PATH, 'rb') as f:
                return pickle.load(f)
        except: pass
    return None

def match_template(user_input, df, models):
    analysis = analyze_input(user_input)
    
    tfidf_vectorizer = models['tfidf_vectorizer']
    count_vectorizer = models.get('count_vectorizer')
    lda_model = models.get('lda_model')
    embeddings = models['embeddings']
    
    similarity_models = load_similarity_models()
    classifier = similarity_models.get('classifier') if similarity_models else None
    
    similarities = {
        'classifier': np.zeros(len(df)),
        'tfidf': np.zeros(len(df)),
        'semantic': np.zeros(len(df)),
        'topic': np.zeros(len(df)),
        'sentiment': np.zeros(len(df))
    }
    
    if classifier is not None:
        try:
            if hasattr(classifier, 'predict_proba'):
                probas = classifier.predict_proba([analysis['processed_text']])[0]
                classes = classifier.classes_
                class_to_proba = dict(zip(classes, probas))
                
                for i, row in df.iterrows():
                    image_name = row['image_name']
                    if image_name in class_to_proba:
                        similarities['classifier'][i] = class_to_proba[image_name]
        except: pass
    
    try:
        user_tfidf = tfidf_vectorizer.transform([analysis['processed_text']])
        if 'tfidf_vector' in df.columns:
            tfidf_vectors = [v for v in df['tfidf_vector'].values if v is not None]
            similarities['tfidf'] = cosine_similarity(user_tfidf, tfidf_vectors).flatten()
        else:
            all_text_tfidf = tfidf_vectorizer.transform(df['processed_text'].values)
            similarities['tfidf'] = cosine_similarity(user_tfidf, all_text_tfidf).flatten()
    except: pass
    
    for i, template_vector in enumerate(embeddings):
        if template_vector is not None:
            similarities['semantic'][i] = cosine_similarity(
                analysis['vector'].reshape(1, -1), 
                template_vector.reshape(1, -1)
            )[0][0]
    
    if lda_model is not None and count_vectorizer is not None and 'topic_distribution' in df.columns:
        try:
            user_count_vector = count_vectorizer.transform([analysis['processed_text']])
            user_topic_dist = lda_model.transform(user_count_vector)[0]
            
            for i, template_topic_dist in enumerate(df['topic_distribution']):
                if template_topic_dist is not None:
                    similarities['topic'][i] = 1 - np.linalg.norm(user_topic_dist - template_topic_dist)
        except: pass
    
    if 'sentiment_score' in df.columns:
        similarities['sentiment'] = 1 - np.abs(df['sentiment_score'] - analysis['sentiment']) / 2
    
    weights = {
        'classifier': 0.3 if np.max(similarities['classifier']) > 0 else 0,
        'tfidf': 0.4 if np.max(similarities['tfidf']) > 0 else 0,
        'semantic': 0.3 if np.max(similarities['semantic']) > 0 else 0,
        'topic': 0.2 if np.max(similarities['topic']) > 0 else 0,
        'sentiment': 0.1 if np.max(similarities['sentiment']) > 0 else 0
    }
    
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        weights = {k: v/weight_sum for k, v in weights.items()}
    else:
        weights = {'tfidf': 0.5, 'semantic': 0.5}
    
    combined_scores = sum(weights[key] * similarities[key] for key in weights)
    combined_scores = np.nan_to_num(combined_scores)
    
    num_results = min(5, len(combined_scores))
    top_indices = np.argsort(combined_scores)[-num_results:][::-1]
    best_idx = top_indices[0]
    best_match = df.iloc[best_idx]
    
    match_info = {
        'classifier_score': similarities['classifier'][best_idx],
        'tfidf_score': similarities['tfidf'][best_idx],
        'semantic_score': similarities['semantic'][best_idx],
        'topic_score': similarities['topic'][best_idx],
        'sentiment_score': similarities['sentiment'][best_idx],
        'combined_score': combined_scores[best_idx],
        'top_templates': df.iloc[top_indices],
        'top_scores': combined_scores[top_indices]
    }
    
    return best_match, match_info