from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

class SurrogateModel():

    def __init__(self):
        self.model = GP(kernel=Matern(nu=1.5),
                        n_restarts_optimizer=10)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X, return_std=False):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, return_std=return_std)
