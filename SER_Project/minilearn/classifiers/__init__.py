from .knn import KNearestNeighbors
from .logistic_regression import LogisticRegression
from .naive_bayes import GaussianNaiveBayes
from .svm import SVM

KNN = KNearestNeighbors

__all__ = [
    "GaussianNaiveBayes",
    "KNearestNeighbors",
    "KNN",
    "LogisticRegression",
    "SVM",
]
