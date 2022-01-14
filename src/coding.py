import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from pprint import pprint
import random

MAX_LAYERS = 5
MAX_NEURONS = 200


class Encoding():

    activation_types = [
        "sigmoid",
        "relu",
        "tanh",
        "hard_sigmoid",
        "linear"
    ]

    def __init__(self, inputs, outputs):
        # layer size, activation type, dropout
        self.inputs = inputs
        self.outputs = outputs
        self.layer_code_size = 1 + len(self.activation_types) + 1

    def num_params(self, x):
        # net = self.decode(x)
        # return net.count_params()
        previous = self.inputs
        count = 0
        for layer_code in self.layer_codes(x):
            size = round(layer_code[0] * MAX_NEURONS)
            if size > 0:
                count += size
                count += previous * size
                previous = size
        count += previous * self.outputs
        return count
        
    def size(self):
        return MAX_LAYERS * self.layer_code_size

    def layer_codes(self, code):
        for start in range(0, MAX_LAYERS*self.layer_code_size, self.layer_code_size):
            yield code[start:start+self.layer_code_size]

    def decode_activation_type(self, act_code):
        return self.activation_types[np.argmax(act_code)]

    def decode(self, code):

        net = Sequential()

        net.add(InputLayer(self.inputs))

        for layer_code in self.layer_codes(code):
            size = round(layer_code[0] * MAX_NEURONS)
            act_type = self.decode_activation_type(layer_code[1:-1])
            dropout = layer_code[-1]

            if size > 0:
                net.add(Dense(size, activation=act_type))
                if dropout > 1e-08:
                    net.add(Dropout(dropout))

        net.add(Dense(self.outputs, activation="softmax"))

        return net

    def encode(self, layer_list):
        code = np.zeros(MAX_LAYERS*self.layer_code_size, dtype=float)
        idx = 0
        for size, activation, dropout in layer_list:
            code[idx] = size/MAX_NEURONS
            idx += 1
            act_type = self.activation_types.index(activation)
            code[idx+act_type] = 1
            idx += len(self.activation_types)
            code[idx] = dropout
            idx += 1
        return code

    def random_code(self):
        code = np.zeros(MAX_LAYERS*self.layer_code_size, dtype=float)

        network = []
        num_layers = random.randrange(1, MAX_LAYERS)
        for _ in range(num_layers):
            size = random.randrange(1, MAX_NEURONS)
            act_type = random.choice(self.activation_types)
            dropout = random.random()

            network.append((size, act_type, dropout))

        return self.encode(network)


if __name__ == "__main__":

    network = [
        (4, "sigmoid", 0),
        (10, "relu", 0.25),
        (20, "relu", 0.25),
        (10, "tanh", 0.3)
    ]

    pprint(network)

    encoding = Encoding(128, 10)

    code = encoding.encode(network)

    net = encoding.decode(code)

    net.summary()
    print(net.to_json())

    code2 = encoding.random_code()
    print(code2)
    net2 = encoding.decode(code2)
    net2.summary()
