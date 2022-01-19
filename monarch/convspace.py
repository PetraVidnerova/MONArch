import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Conv2D, MaxPooling2D, Flatten

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
    # crossover conv part
    c1, c2 = a.body[0], b.body[0] 
    if len(c1) == 0:
        point1 = 0
    else:
        point1 = random.randrange(0, len(c1)+1)
    if len(c2) == 0:
        point2 = 0
    else:
        point2 = random.randrange(0, len(c2)+1)

    conv_part1 = c1[:point1] + c2[point2:]
    conv_part2 = c2[:point2] + c1[point1:]
    
    # crossover dense part 
    a, b = a.body[1], b.body[1] 
    if len(a) == 0:
        point1 = 0
    else:
        point1 = random.randrange(0, len(a)+1)
    if len(b) == 0:
        point2 = 0
    else:
        point2 = random.randrange(0, len(b)+1)
    dense_part1 = a[:point1] + b[point2:]
    dense_part2 = b[:point2] + a[point1:]

    return [
        Individual([conv_part1, dense_part1]),
        Individual([conv_part2, dense_part2]) 
    ]


def mutate(a):
    conv_part, dense_part = a.body
    if random.random() < 0.5:
        # mutate conv
        a = conv_part
        if len(a) == 0:
            return Individual([conv_part, dense_part])
        layer = random.randrange(len(a))
        what = random.randrange(4)
        if what == 0: # number of filters
            new_size = a[layer][0] + random.choice(range(-10,10))
            if new_size <= 0:
                new_layer = None
            else:
                new_layer = (new_size, a[layer][1], a[layer][2], a[layer][3])
        elif what == 1: # filter size
            kernel_size = a[layer][1] + random.choice(range(-2,2))
            if kernel_size < 2:
                kernel_size = 2
            new_layer = (a[layer][0], kernel_size, a[layer][2], a[layer][3])
        elif what == 2: # activation
            new_act = random.choice(ACTIVATION_TYPES)
            new_layer = (a[layer][0], a[layer][1], new_act, a[layer][3])
        elif what == 3: # pool size
            pool_size = a[layer][3] + random.choice(range(-1, 1))
            if pool_size <= 1:
                pool_size = 0
            new_layer = (a[layer][0], a[layer][1], a[layer][2], pool_size)
        child = []
        for i in range(layer):
            child.append(a[layer])
        if new_layer is not None:
            child.append(new_layer)
        for i in range(layer+1, len(a)):
            child.append(a[layer])
        ret = [child, dense_part]
            

    else:
        # mutate dense
        a = dense_part
        if len(a) == 0:
            return Individual([conv_part, dense_part])
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
        ret = [conv_part, child]
        
    return Individual(ret)
        
        

class Space:

    def __init__(self, inputs, outputs,
                 max_conv_layers=5,
                 min_filters=10,
                 max_filters=100,
                 min_kernel_size=2,
                 max_kernel_size=5,
                 min_pool_size=2,
                 max_pool_size=3,
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
        pool_layer = random.random() < 0.3
        pool_size = random.randrange(min_pool_size, max_pool_size) if pool_layer else 0
        return [filters, kernel, activation_type, pool_size]
        
        
    def get_random_sample(self):
        conv_part = []
        
        num_conv_layer = random.randrange(1, self.max_conv_layers+1)
        conv_part = [
            self.random_conv_layer(self.min_filters, self.max_filters,
                                   self.min_kernel_size, self.max_kernel_size,
                                   self.min_pool_size, self.max_pool_size)
            for _ in range(num_conv_layer)
        ]
        
        dense_part = []
        num_layers = random.randrange(1, self.max_layers+1)
        for _ in range(num_layers):
            size = random.randrange(10, self.max_neurons)
            act_type = random.choice(self.activation_types)
            dropout = 0.5 * random.random()

            dense_part.append((size, act_type, dropout))

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
                       padding='same',
                       activation=activation)
            )
#            net.add(Activation(activation))               
            if pool_size > 0:
                try:
                    net.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
                except ValueError:
                    print("Incorect pooling")
                    return None
 
        net.add(Flatten())
        for size, activation, dropout in dense_part:
            net.add(Dense(size, activation=activation))
            if dropout > 0:
                net.add(Dropout(dropout))

        net.add(Dense(self.outputs, activation="softmax"))
                
        return net

    # def num_params(self, x):
    #     previous = self.inputs
    #     count = 0
    #     for size, _, _ in x:
    #         if size > 0:
    #             count += size
    #             count += previous * size
    #             previous = size
    #     count += previous * self.outputs
    #     return count

    
    def get_features(self, x):

        conv_part, dense_part = x

        net = self.create_network(x)
        param_count = net.count_params()//1000 if net is not None else 100000 

        # conv part 
        num_conv_layers = len(conv_part)
        num_pool_layers = 0
        for l in conv_part:
            if l[3] > 0:
                num_pool_layers += 1
        mean_filter_size = 0
        for l in conv_part:
            mean_filter_size += l[1]
        mean_filter_size /= len(conv_part)

        act_counts = { a:0 for a in self.activation_types }
        for _, act_type, _ in dense_part:
            act_counts[act_type] += 1
        count = len(conv_part)
        if count != 0:
            for key in act_counts:
                act_counts[key] /= count

        
        ret_conv = [num_conv_layers, num_pool_layers,
                    mean_filter_size,
                    *act_counts.values()]

        # dense part 
        num_layers = len(dense_part)

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
            
        ret_dense = [num_layers,  *list(act_counts.values()), *dropout]
        
        return np.array([param_count]+ret_conv+ret_dense)
        
    
    def nfeatures(self):
        return 3 + 2 + 2*len(self.activation_types) + 3
