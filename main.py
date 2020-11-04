
import pandas as pd
import numpy as np

from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB

from vectorizer import strip_stopwords, get_vectorizer
from classifier import get_classifier

import os

def preprocess_line_(line):
    for w in line.split(' '):
        # all caps has special meaning
        if w.upper() == w:
            pass
        else:
            line = line.replace(w, w.lower())

    if strip_stopwords is True:
        return reduce(lambda line, sw: line.replace(sw, ''), stopwords, line)
    else:
        return line


def preprocess(data):
    data['message'] = data['message'].apply(preprocess_line_)
    return data


def load_data(filename, test=False, **kwargs):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', filename), **kwargs)
    return preprocess(data)


def transformer(classifier, vectorizer):
    return lambda message: classifier.predict(vectorizer.transform([message]).toarray())


def train(train_data, should_ignore_cases=True):

    vectorizer = get_vectorizer()
    vectorizer.fit(train_data['message'].to_numpy())
    features_count = len(vectorizer.get_feature_names())

    raw_train_data = train_data['message'].to_numpy()

    X = vectorizer.transform(raw_train_data).toarray()
    Y = train_data['label'].to_numpy()

    classifier = get_classifier()
    classifier.fit(X, Y)

    return {
        'classifier': classifier,
        'vectorizer': vectorizer,
    }


def test(test_data, classifier, vectorizer, is_debug=True):
    # tr = transformer(classifier, vectorizer)

    raw_test_data = test_data['message'].to_numpy()
    X = vectorizer.transform(raw_test_data).toarray()

    Y = classifier.predict(X)

    if is_debug == True:
        from debug import debug
        debug(raw_test_data, load_data, Y)

    test_result = pd.DataFrame(Y, columns=['label'])
    test_result.insert(0, 'id', test_data['id'])
    return test_result


def main():
    data = load_data('train.csv')
    model = train(data)

    test_data = load_data('leaderboard_test_file.csv', test=True)

    test_labels = test(test_data, **model)
    test_labels.to_csv('out.csv', index=False)

if __name__ == '__main__':
    main()
