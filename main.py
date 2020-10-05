
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

import os


def load_data(filename, test=False):
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', filename))


def transformer(classifier, vectorizer):
    return lambda message: classifier.predict(vectorizer.transform([message]).toarray())


def train(train_data):
    stopwords = [',', '.', ' ']

    vectorizer = CountVectorizer(analyzer='word', decode_error='strict',
            encoding='utf-8', input='content',
            lowercase=False, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), stop_words=stopwords,
            strip_accents='unicode')

    vectorizer.fit(train_data['message'].to_numpy())
    features_count = len(vectorizer.get_feature_names())

    raw_train_data = train_data['message'].to_numpy()

    X = vectorizer.transform(raw_train_data).toarray()
    Y = train_data['label'].to_numpy()

    classifier = GaussianNB()
    classifier.fit(X, Y)

    return {
        'classifier': classifier,
        'vectorizer': vectorizer,
    }


def test(test_data, classifier, vectorizer):

    tr = transformer(classifier, vectorizer)

    raw_test_data = test_data['message'].to_numpy()
    X = vectorizer.transform(raw_test_data).toarray()

    Y = classifier.predict(X)

    return pd.DataFrame(Y, columns=['label'])


def main():
    data = load_data('train.csv')
    model = train(data)

    test_data = load_data('leaderboard_test_file.csv', test=True)

    test_labels = test(test_data, **model)
    test_labels.to_csv('out.csv')

if __name__ == '__main__':
    main()
