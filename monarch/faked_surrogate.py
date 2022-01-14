import numpy as np

class DummyModel():

    def __init__(self, encoding):
        self.e = encoding

    def fit(self, X, y):
        pass

    def predict(self, X):
        y = [
            -x[1]
            for x in X
        ]
        return np.array(y)
