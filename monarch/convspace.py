import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout

ACTIVATION_TYPES = [
    "sigmoid",
    "relu",
    "tanh",
    "hard_sigmoid",
    "linear"
]

class Individual():


    def __init__(self, code=None):
        if code is None:
            self.body = []
        else:
            self.body = code


def crossover(a, b):
    a, b = a.body, b.body 
    if len(a) == 0:
        point1 = 0
    else:
        point1 = random.randrange(0, len(a)+1)
    if len(b) == 0:
        point2 = 0
    else:
        point2 = random.randrange(0, len(b)+1)
    return [
        Individual(a[:point1] + b[point2:]),
        Individual(b[:point2] + a[point1:]) 
    ]


def mutate(a):
    a = a.body 
    if len(a) == 0:
        return a
    layer = random.randrange(len(a))
    what = random.randrange(3)
    if what == 0:
        new_size = a[layer][0] + random.choice(range(-10,10))
        if new_size <= 0:
            new_layer = None
        else:
            new_layer = (new_size, a[layer][1], a[layer][2])
    elif what == 1:
        new_act = random.choice(ACTIVATION_TYPES)
        new_layer = (a[layer][0], new_act, a[layer][2])
    else:
        new_dropout = a[layer][2] + (-0.1 + (np.random.rand()*0.2))
        if new_dropout <  0.0:
            new_dropout = 0
        elif new_dropout > 0.5:
            new_dropout = 0.5
        new_layer = (a[layer][0], a[layer][1], new_dropout)
    child = []
    for i in range(layer):
        child.append(a[layer])
    if new_layer is not None:
        child.append(new_layer)
    for i in range(layer+1, len(a)):
        child.append(a[layer])
    return Individual(child)
        
        

class Space:

    def __init__(self, inputs, outputs,
                 max_conv_layers=5,
                 min_filters=10,
                 max_filters=100,
                 min_kernel_size=2,
                 max_kernel_size=5,
                 min_pool_size=2,
                 max_pool_size=4,
                 max_layers=5, max_neurons=1000,
                 activation_types=ACTIVATION_TYPES):
        self.inputs = inputs
        self.outputs = outputs
        self.max_conv_layers = max_conv_layers
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.activation_types = activation_types

    def random_conv_layer(self, min_filters, max_filters, min_kernel_size, max_kernel_size, min_pool_size, max_pool_size):
        filters = random.randrange(min_filters, max_filters+1)
        kernel = random.randrange(min_kernel_size, max_kernel_size)
        activation_type = random.choice(ACTIVATION_TYPES)
        # flip pool coin
        pool_layer = random.rand() < 0.7
        pool_size = random.randrange(min_pool_size, max_pool_size) if pool_layer else 0
        return [filters, kernel, activation_type, pool_size]
        
        
    def get_random_sample(self):
        conv_part = []
        
        num_conv_layer = random.randrange(1, self.max_conv_layers+1)
        conv_part = [
            random_conv_layer(min_filters, max_filters, min_kernel_size, max_kernel_size, min_pool_size, max_pool_size)
            for _ in range(num_conv_layer)
        ]
        
        dense_part = []
        num_layers = random.randrange(1, self.max_layers+1)
        for _ in range(num_layers):
            size = random.randrange(10, self.max_neurons)
            act_type = random.choice(self.activation_types)
            dropout = 0.5 * random.random()

            network.append((size, act_type, dropout))

        return [conv_part, dense_part]

    # def select_and_crossover(self, individuals, scores, number):

    #     probs = scores.flatten()
    #     probs /= np.sum(probs)
    #     idx = np.random.choice(len(individuals), size=number+(number%2), p=probs)
    #     parents = [ individuals[i] for i in idx ]
    #     childs = []
    #     for i in range(number//2):
    #         childs.extend(crossover(parents[2*i],
    #                                 parents[2*i+1]))
    #     childs = [ mutate(a) for a in childs ]
    #     return childs 

    def create_network(self, x):
        net = Sequential()
        net.add(InputLayer(self.inputs)) 

        conv_part, dense_part = x 
        
        for filters, kernel_size,  activation, pool_size in conv_part:
            net.add(                                                                                                                
                Conv2D(filters,
                       (kernel_size, kernel_size),
                       padding='same')
            )
            net.add(MaxPooling2d(pool_size=(pool_size, pool_size)))
            net.add(Activation(activation))               

        net.add(Flatten())
        for size, activation, dropout in dense_part:
            net.add(Dense(size, activation=activation))
            if dropout > 0:
                net.add(Dropout(dropout))

        net.add(Dense(self.outputs, activation="softmax"))
                
        return net

    def num_params(self, x):
        previous = self.inputs
        count = 0
        for size, _, _ in x:
            if size > 0:
                count += size
                count += previous * size
                previous = size
        count += previous * self.outputs
        return count

    
    def get_features(self, x):

        conv_part, dense_part = x

        param_count = ... 
        ...
        ret_conv = [num_conv_layers, num_pooling_layers, param_count,
                    mean_filter_size,
                    *act_counts.values()]
        
        num_layers = len(dense_part)
        param_count = self.create_network(dense_part).count_params()

        act_counts = { a:0 for a in self.activation_types }
        for _, act_type, _ in dense_part:
            act_counts[act_type] += 1

        count = len(dense_part)
        if count != 0:
            for key in act_counts:
                act_counts[key] /= count

        drop_rates = [ r for _, _, r in dense_part ]     
        if not drop_rates:
            dropout = (0,0,0)
        else:
            dropout = (
                min(drop_rates),
                max(drop_rates),
                sum(drop_rates)/len(drop_rates) 
            )
            
        ret_dense = [num_layers, param_count, *list(act_counts.values()), *dropout]
        
        return np.array(ret_conv+ret_dense)
        
    
    def nfeatures(self):
        return 2 + len(self.activation_types) + 3
