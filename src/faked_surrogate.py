import numpy as np

class DummyModel():

    def __init__(self, encoding):
        self.e = encoding

    def fit(self, X, y):
        pass

    def predict(self, X):
        y = [
            - self.e.num_params(x) // 1000
            for x in X
        ]
        return np.array(y)
