from functools import partial
from tensorflow.keras.datasets import mnist 
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

import mobopt as mo

from search import NeMOBOptimizer
from space import Space, Individual, mutate, crossover
from objective import objective, crossval_objective

from faked_surrogate import DummyModel
from surrogate import SurrogateModel

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

def create_random_from_space(space, indclass): 
    individual = indclass()
    individual.body = space.get_random_sample()
    return individual
create_random = partial(create_random_from_space, space)

#
# create optimizer 
#
optimizer = mo.MOBayesianOpt(target=objective_func,
                             NObj=2,
                             verbose=True,
                             max_or_min='max',
                             random_points=(Individual,create_random,crossover,mutate), 
                             predictors=[SurrogateModel(), DummyModel(space)],
                             getfeatures=space.get_features,
                             nfeatures=space.nfeatures()
                             )

optimizer.initialize(init_points=6)

front, pop = optimizer.maximize(n_iter=200)

objective_values = optimizer.y_Pareto
individuals = optimizer.x_Pareto

print(objective_values)

print(objective_values.argmax(axis=0))

i = objective_values.argmax(axis=0)[0]

print(i)

best = individuals[i] 


print("final learning")
objs = []
for i in range(5):
    objs.append(objective(space, (X_train, y_train, X_test, y_test), best.body, epochs=20, stopping=False))

print(f"Results: {objs}")
print(f"Results mean: {sum(objs)/len(objs)}")

