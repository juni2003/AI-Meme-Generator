"""Text preprocessing functions"""
import re, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

BASIC_STOPWORDS = set([
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
])

def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except:
        return BASIC_STOPWORDS

def simple_lemmatize(word):
    if len(word) < 4:
        return word
    
    # Plurals
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('es') and len(word) > 3:
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and not word.endswith('us'):
        return word[:-1]
    
    # Verbs
    if word.endswith('ing') and len(word) > 5:
        if word[-4] == word[-5]:
            return word[:-4]
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        if word[-3] == word[-4]:
            return word[:-3]
        return word[:-2]
    
    # Adjectives
    if word.endswith('er') and len(word) > 4:
        return word[:-2]
    if word.endswith('est') and len(word) > 5:
        return word[:-3]
    
    return word

def enhanced_preprocess_text(text):
    if text is None or text == "":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    try:
        tokens = word_tokenize(text)
    except:
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [word for word in text.lower().split() if word]
    
    stop_words = get_stopwords()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [simple_lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if len(word) > 2]
    
    return ' '.join(tokens)