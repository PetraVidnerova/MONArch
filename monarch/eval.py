from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils

import numpy as np 
import json
import sys

from objective import objective
from convspace import Space

def convert_data(X, y):

    X  = X[..., np.newaxis]
    X = X.astype("float32")
    X /= 255

    y = utils.to_categorical(y)

    return X, y

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train,  y_train = convert_data(X_train, y_train)
X_test, y_test = convert_data(X_test, y_test)


name = sys.argv[1]


with open(f"{name}.json","r") as f:
    objectives, networks = json.load(f)
    

results = zip(objectives, networks)

space  = Space((28,28,1), 10)


for obj, net in results:
    print("RES:", obj)
    print("final learning")

    objs = []
    for i in range(5):
        objs.append(objective(space, (X_train, y_train, X_test, y_test), np.array(net), epochs=20, stopping=False))

    print(f"Results: {objs}")
    print(f"RES: Results mean: {sum(objs)/len(objs)}")
    print()
