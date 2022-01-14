from functools import partial
from tensorflow.keras.datasets import mnist 
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

from search import NeMOBOptimizer
from space import Space
from objective import objective, crossval_objective

#
# prepare data
#
def convert_data(X, y):
    X = X.reshape(X.shape[0], -1)
    X = X.astype("float32")
    X /= 255

    y = utils.to_categorical(y)

    return X, y

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train,  y_train = convert_data(X_train, y_train)
X_test, y_test = convert_data(X_test, y_test)

X_t, X_val, y_t, y_val = train_test_split(X_train, y_train)

D = (X_t, y_t, X_val, y_val)

#
# create search space 
#
space = Space(784, 10)

#
# bound objective function to space and data 
#
#objective_func = partial(objective, space, D)
objective_func = partial(crossval_objective, space, (X_train, y_train))
#
# create optimizer 
#
optimizer = NeMOBOptimizer(objective_func,
                           space
)


#
# run optimisation 
#
res = optimizer.optimize(n_steps=50)

print("final learning")
objs = []
for i in range(5):
    objs.append(objective(space, (X_train, y_train, X_test, y_test), res, epochs=20, stopping=False))

print(f"Results: {objs}")
print(f"Results mean: {sum(objs)/len(objs)}")

