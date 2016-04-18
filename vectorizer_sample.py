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


class TestSimpleVectorizer(TestCase):
    def test_unittest0(self):
        pass

    def test_unittest1(self):
        regr = linear_model.LinearRegression()

        x_train = ["", "", ""]
        y_train = [1, 2, 3, 40]

        x_test = ["x_tescxvvzvt"]

        combined_features = FeatureUnion([("sv1", SimpleVectorizerExample("SV1")),
                                          ("sv2", SimpleVectorizerExample("SV2"))])
        pipeline = Pipeline([('union', combined_features), ('rerg', regr)])
        pipeline.fit(x_train, y_train)

        y_test = pipeline.predict(x_test)
        print ("prediction result")
        print(y_test)

        self.assertTrue(y_test.__len__() == 4)