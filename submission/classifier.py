
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

def get_classifier():
    # SVM + SGD
    return SGDClassifier(alpha=.0001, max_iter=50, penalty='l2')
