import numpy as np
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Conv1D, merge, concatenate, MaxPooling1D, GlobalMaxPooling1D, 
Embedding, Dropout, Activation, SeparableConv2D, Conv2D, BatchNormalization, Flatten, GlobalMaxPooling2D, 
                          GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D, Layer, PReLU)
from keras.optimizers import Nadam, Adam
from keras.layers.core import Reshape
import database as db
from factor_tools import sorted_factors, most_common_factors, middle_factors
from os.path import join
import re
import sys
from IPython.core.debugger import set_trace
import pickle
from keras import backend as K
from copy import copy

def get_primitives():
    return [
    '', 'identity', 'batchnormalization', 'dense', 'activation', 'conv2d1x1', 'separableconv2d3x3', 'maxpooling2d3x3', 'averagepooling2d3x3', 'separableconv2d1x1', 'dropout', 'conv2dsep1x1', 'conv2d3x3', 'conv2dsep3x3']

def AvgPlusMaxPooling2D(x):
    max_pool = MaxPooling2D(pool_size = (3, 3), strides = None, padding = "same")(x)
    avg_pool = AveragePooling2D(pool_size = (3, 3), strides = None, padding = "same")(x)
    return Add()([max_pool, avg_pool])
    
def AvgPlusMaxPooling2DOutputShape(input_shape):
    shape = list(input_shape)
    shape[1] //= 3
    shape[2] //= 3
    return tuple(shape)

class Antirectifier(Layer):
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=1)
        pos = K.relu(inputs)
        neg = K.relu(-inputs)
        return K.concatenate([pos, neg], axis=1)
    
def zero_pad_layers(layers):
    max_w = 0
    max_h = 0
    for layer in layers:                                             
        if layer.shape[1].value is not None:
            max_w = max(max_w, layer.shape[1].value)
            if len(layer.shape) > 2:
                 max_h = max(max_h, layer.shape[2].value)
            else:
                max_h = copy(max_w)
           
    for k, layer in enumerate(layers):
        if layer.shape[1].value is not None:
            w_diff = max_w - layer.shape[1].value
            if len(layer.shape) > 2:
                h_diff = max_h - layer.shape[2].value
            else:
                h_diff = copy(w_diff)

            if w_diff > 0 or h_diff > 0:
                layers[k] = ZeroPadding2D(((int(np.ceil(w_diff/2)), int(np.floor(w_diff/2))), 
                                                  (int(np.ceil(h_diff/2)), int(np.floor(h_diff/2)))))(layer)
    return layers

def reshape_layers(layers):
    print("Attempting to reshape layer, not tested")
    factors_pool = []
                           
    for layer in layers:
        for i in range(len(layer.shape) - 1):
            j = i + 1
            factors_pool = factors_pool + sorted_factors(layer.shape[j].value)
        common_factor_x, common_factor_y = most_common_factors(factors_pool)
        print("Reshaped", "common_factor_x", "common_factor_y", common_factor_x, common_factor_y)
           
    for k, layer in enumerate(layers):
        layers[k] = Reshape((common_factor_x, common_factor_y, -1))(layer)
    return layers



from keras.layers import Input
from copy import copy, deepcopy

def convert_graph_to_keras(G, inputs):
#     data = db.select("""SELECT rowid, motif_filename FROM motifs WHERE ready = 1""")
#     motifs = {}
#     try:
#         for motif in data:
#             with open(motif[1], 'rb') as f:
#                 motifs[str(motif[0])] = pickle.load(f)
#     except FileNotFoundError as e:
#         print(data)
#         print(e)

    graphMapping = dict({"0": "", "1": "conv2d1x1", "2": "conv2dsep1x1", "3": "conv2d3x3", "4": "conv2dsep3x3"})
     
    G = deepcopy(G)
    for i in range(len(G)):
        for j in range(len(G[0])):
            if j > i and j < i + 3:
                G[i][j] = graphMapping[str(int(G[i][j]))]
            else:
                G[i][j] = ""
    
    #G.append([])
    #for i in range(len(G[1])):
     #   G[-1].append('')
        
    node_inputs = []

    for i in range(len(G[1])):
        node_inputs.append([])
    node_inputs[0].append(inputs)
    
    curr_output = inputs
    
    #node_inputs = [
    #    [], [], [],
    #]
    
    #[
    #    ['', 'conv2d', ''],
    #    ['', '', 'averagepooling2d'],
    #    ['', '', '']
    #]

    for i in range(len(G)):
        if len(node_inputs[i]) > 1:
            zero_pad_layers(node_inputs[i])
            node_inputs[i] = [concatenate(node_inputs[i], axis = -1)]
            curr_output = node_inputs[i][0]
        for j in range(len(G[0])):
            if len(G[i][j]) > 0:
                if G[i][j].isdigit():
                    curr_output = convert_graph_to_keras(motifs[G[i][j]], inputs = node_inputs[i][0])
                    node_inputs[j].append(curr_output)
                    if node_inputs[i][0] == inputs:
                        node_inputs[i] = [curr_output]
                else:
#                 elif len(node_inputs[i]) > 0:
                    curr_output = convert_to_keras_layer(G[i][j], prev = curr_output)
                    node_inputs[j].append(curr_output)
                    if len(node_inputs[i]) > 0 and node_inputs[i][0] == inputs:
                        node_inputs[i] = [curr_output]
                        
    return curr_output


p = re.compile("ndim=[0-9]*")

def reshape_layer(layer, flatten = False):
    #dims = [m[5:] for m in p.findall(str(e))]
    #target_shape = ()
    #for i in range(np.abs(dims[1] - dim[0])):
    #    target_shape += (0,)
    
    factors_pool = []
    
    prod_sum = 1
    for i in range(len(layer.shape) - 1):
        j = i + 1
        prod_sum *= layer.shape[j].value
        
    if flatten:
        return Reshape((-1,))(layer)
        
    factors = middle_factors(prod_sum)
    
    reshaped_layer = Reshape((factors[0], factors[1], -1))(layer)
    return reshaped_layer

def convert_to_keras_layer(layer_type, prev, num_filters = 16, dense_size = 256, max_params = 1e6, no_repeat = False):
    max_params = max_params // dense_size
    layer_type = layer_type.lower()
    
    try:
        if layer_type == 'identity':
            return prev
        elif layer_type == 'conv2dtranspose3x3':
            x = layer.Conv2DTranspose(num_filters, 3, padding = 'same')(prev)
            return x
        elif layer_type == 'batchnormalization':
            x = BatchNormalization()(prev)
            return x
        elif layer_type == 'dense':
            if prev.shape[1].value is not None:
                prod_sum = 1
                for i in range(len(prev.shape) - 1):
                    j = i + 1
                    prod_sum *= prev.shape[j].value
                if prod_sum > max_params:
                    prev = convert_to_keras_layer('averagepooling2d3x3', prev)
                    return convert_to_keras_layer('dense', prev)
                if len(prev.shape) > 2:
                    prev = reshape_layer(prev, flatten = True)
            x = Dense(dense_size)(prev)
            return x
        elif layer_type == 'dropout':
            x = Dropout(0.5)(prev)
            return x
        elif layer_type == 'activation':
            if len(prev.shape) == 2:
                x = Antirectifier()(prev)
            else:
                x = PReLU()(prev)
            return x
        elif layer_type == 'conv2d1x1':
            x = Conv2D(num_filters, 1, padding = 'same')(prev)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        elif layer_type == 'conv2d3x3':
            x = Conv2D(num_filters, 3, padding = 'same')(prev)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        elif layer_type == 'conv2dsep1x1':
            x = SeparableConv2D(num_filters, 1, padding = 'same')(prev)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        elif layer_type == 'conv2dsep3x3':
            x = SeparableConv2D(num_filters, 3, padding = 'same')(prev)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        elif layer_type == 'maxpooling2d3x3':
            x = MaxPooling2D(3, padding = 'same')(prev)
            return x
        elif layer_type == 'averagepooling2d3x3':
            x = AveragePooling2D(3, padding = 'same')(prev)
            return x
        elif layer_type == 'avgplusmaxpooling2d3x3':
            x = Lambda(AvgPlusMaxPooling2D, output_shape = AvgPlusMaxPooling2DOutputShape)(prev)
            return x
        else:
            print("Layer type not found")
            print(layer_type)
            sys.exit(1)
    except ValueError as e:
        if no_repeat:
            print(e)
            raise Exception("Reshape failed, layer_type is {}".format(layer_type))
        reshaped_prev = reshape_layer(prev)
        return convert_to_keras_layer(layer_type, reshaped_prev, no_repeat = True)


# def load_weights(model, weights):
#     for layer in model.layer()
#         set_trace
#         test = "hi"
    
def assemble_model(adj_matrix,
                   input_shape = (32, 32, 3), 
                   optim = Adam(lr = 5e-4), 
                   num_outputs = 10,
                   output_type = 'categorical',
                   max_params = 5e6,
                   weights = None):
    inputs = Input(shape = input_shape)
    
    x = convert_graph_to_keras(adj_matrix, inputs)
    
    if len(x.shape) != 2:
        x = Flatten()(x)
        
#     x = convert_to_keras_layer("dense", x)
#     x = BatchNormalization()(x)
#     x = Antirectifier()(x)
#     x = Dropout(.5)(x)
        
    if output_type is 'categorical':
        outputs = Dense(num_outputs, activation = 'softmax')(x)
        model = Model(inputs = inputs, outputs = outputs)
        model.compile(optimizer=optim,
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    else:
        outputs = Dense(num_outputs)(curr_output)
        model = Model(inputs = inputs, outputs = outputs)
        model.compile(optimizer=optim,
                  loss='mse',
                  metrics=['mae'])
    
    if model.count_params() > max_params:
        print(model.count_params())
        raise Exception("Too many parameters")
    
#     if weights is not None:
#         load_weights(model, weights)
    
    return model

from clr_callback import CyclicLR
from keras.optimizers import SGD

# 4-6 epochs seems to work the best, maybe 2 fine tune (halved lr) and 4 normal
def fitWithCLR(model, ds, epochs = 4, optim = SGD(nesterov=True), finetune = False):
    base_lr = 0.001
    max_lr = 0.006
    if finetune:
        base_lr /= 2
        max_lr /= 2
        epochs //= 2
    clr = CyclicLR(base_lr=base_lr, max_lr=max_lr,
                                step_size=2000., mode='triangular')
    
    model.compile(optimizer = optim,
         loss = "categorical_crossentropy",
         metrics = ["acc"])
    
    return model.fit_generator(generator=ds.train_generator, steps_per_epoch=ds.train_steps, 
                               epochs = epochs, verbose=1, callbacks=[clr], validation_data=ds.val_generator,
                               validation_steps=ds.val_steps)