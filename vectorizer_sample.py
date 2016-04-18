from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from sklearn import linear_model
from unittest import TestCase

class SimpleVectorizerExample(BaseEstimator):
    def __init__(self, name):
        self._name = name

    def fit(self, x, y=None):
        print(self._name)
        print("SimpleVectorizerExample fit")
        return self

    def transform(self, input_data_set_full, y=None):
        print(self._name)
        print("SimpleVectorizerExample transform")
        feature = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        return [feature, feature, feature, feature]

