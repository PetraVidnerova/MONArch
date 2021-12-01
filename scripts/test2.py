from functools import partial
import numpy as np
import mobopt as mo

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils

from coding import Encoding
from objective import objective

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


def convert_data(X, y):
    X = X.reshape(X.shape[0], -1)
    X = X.astype("float32")
    X /= 255

    y = utils.to_categorical(y)

    return X, y


X_train,  y_train = convert_data(X_train, y_train)
X_test, y_test = convert_data(X_test, y_test)

X_t, X_val, y_t, y_val = train_test_split(X_train, y_train)

print(X_t.shape, y_t.shape)
print(X_val.shape, y_val.shape)

D = (X_t, y_t, X_val, y_val)
e = Encoding(784, 10)

objective_func = partial(objective, e, D)

pbounds = np.array(
    [0, 1]*e.size(),
    dtype="float"
).reshape(-1, 2)

print(pbounds)
print(pbounds.shape)

optimizer = mo.MOBayesianOpt(target=objective_func,
                             NObj=2,
                             pbounds=pbounds,
                             verbose=True,
                             max_or_min='max',
                             random_points_generator=e.random_code
                             )

optimizer.initialize(init_points=6)

front, pop = optimizer.maximize(n_iter=2000)

objective_values = optimizer.y_Pareto
individuals = optimizer.x_Pareto

print(objective_values)

i = objective_values.argmax(axis=1)[0]

print(i)

best = individuals[i] 

print(
    objective(
        e,
        (X_train, y_train, X_test, y_test),
        best,
        stopping=False,
        epochs=100
    )
)
