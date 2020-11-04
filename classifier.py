
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

def get_classifier():
    # return LinearSVC(penalty='l2', dual=False, tol=1e-3)
    return SGDClassifier(alpha=.0001, max_iter=50, penalty='l2')
