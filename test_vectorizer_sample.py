from unittest import TestCase


class TestSimpleVectorizer(TestCase):
    def test_unittest0(self):
        pass

    def test_unittest1(self):
        from sklearn.pipeline import Pipeline, FeatureUnion
        from sklearn import linear_model
        from vectorizer_sample import SimpleVectorizerExample

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