
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

stopwords = [',', '.', '\\', '"', '\'', '*']
strip_stopwords = False

def get_vectorizer():
    kwargs = {
        'analyzer': 'char_wb',
        'decode_error': 'strict',
        'encoding': 'utf-8',
        'input': 'content',
        'stop_words': stopwords,
        'strip_accents': 'unicode',
    }

    return TfidfVectorizer(**kwargs, lowercase=True, min_df=1, sublinear_tf=True, max_df=0.3, ngram_range=(2,6))
