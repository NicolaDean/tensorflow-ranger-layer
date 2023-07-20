import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import pathlib
import sys
import random as rn
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import ResNet50,VGG19
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout,Input,Add, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.engine import functional
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import re
from keras.models import Model


RANGER_MODULE_PATH = "../"

# appending a path
sys.path.append(RANGER_MODULE_PATH) #CHANGE THIS LINE

from custom_layers.ranger import *

def inject_layer_topology(layer,position,topology,math_cond,insert_layer_factory):
        
        print(f"Layer: {layer.name}")
        if isinstance(layer,functional.Functional):
            print(f"FUNCTIOANL RECURSION: {layer.name}")
            for l in layer.layers[1:]:
                x = inject_layer_topology(l,position,topology,math_cond,insert_layer_factory)
            return x
        
        # Determine input tensors
        layer_input = [topology['new_output_tensor_of'][layer_aux] for layer_aux in topology['input_layers_of'][layer.name]]

        if len(layer_input) == 1:
            layer_input = layer_input[0]
        elif len(layer_input) == 2:
            tmp = layer_input[1]
            layer_input[1] = layer_input[0]
            layer_input[0] = tmp

        # Insert layer if name matches the regular expression
        if math_cond(layer):
            print(f"Match for {layer.name}")
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            #Create the layer to insert
            new_layer = insert_layer_factory(layer)

            new_layer._name = '{}_{}'.format(new_layer.name,layer.name)

            #Compute output tensor of new layerqq
            x = new_layer(x)

            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        topology['new_output_tensor_of'].update({layer.name: x})
        return x
    
def explore_model_topology(topology,model):
    #Set the input layers of each layer (Reconstruct model graph)
    for layer in model.layers:
        #Recursion for functional api
        if isinstance(layer,functional.Functional):
            print(f"Recursion on {layer.name}")
            topology = explore_model_topology(topology,layer)

        xxx = len(layer._outbound_nodes)
        print(f"Layer [{layer.name}] => has {len(layer._outbound_nodes)} outputs")
        for node in layer._outbound_nodes:
            input_layer = layer.name
            if isinstance(layer,functional.Functional):
                input_layer = layer.layers[len(layer.layers)-1].name
                print(f"AAAAAAAAAAAAAA: {input_layer}")

            layer_name = node.outbound_layer.name

            if layer_name not in topology['input_layers_of']:
                topology['input_layers_of'].update({layer_name: [input_layer]})
            else:
                topology['input_layers_of'][layer_name].append(input_layer)
                
        #Remove Old connections
        for _ in range(xxx):
            layer._outbound_nodes.pop()
        
    return topology
   
def insert_layer_nonseq(model, math_cond, insert_layer_factory,
                        insert_layer_name=None, position='after'):
    index = 0
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    network_dict = explore_model_topology(network_dict,model)

    finetuning = isinstance(model.layers[0],functional.Functional)

    print(type(model.layers[0]))
    # Set the output tensor of the input layer
    if not finetuning:
        network_dict['new_output_tensor_of'].update({model.layers[0].name: model.input})
    else:
        print("NOT FINE TUNING")
        network_dict['new_output_tensor_of'].update({model.layers[0].layers[0].name: model.layers[0].input})

    print(network_dict)
    # Iterate over all layers after the input
    model_outputs = []
    
    if isinstance(model.layers[0],functional.Functional):
        input_layer = model.layers[0].layers[0].input
        start = 0
        print("Model contain functional API Input")
    else:
        start = 1
        input_layer = model.input
        print("Model has Vanilla input")

    for layer in model.layers[start:]:
        
        x = inject_layer_topology(layer,position,network_dict,math_cond,insert_layer_factory)

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names: #TODO => PROBABLY IT CAN BE DONE RECURSIVLY
            print(f"OUTPUT LAYER => {layer.name}")
            print(f"REPLACED LAYER => {x.name}")
            model_outputs.append(x)

    return Model(inputs=input_layer, outputs=model_outputs)



'''
#EXAMPLE OF USAGE
baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=(50, 50, 3)))

baseModel = tf.keras.models.load_model("../saved_models/vgg19_gtsrb")
baseModel.summary()

def math_cond(layer):
    if isinstance(layer,Conv2D):
        return True
    else:
        return False
    
def ranger_layer_factory(layer):
    return Ranger(name=f"ranger")

#model = insert_layer_nonseq(baseModel, '.*conv*._block.*', dropout_layer_factory)
model = insert_layer_nonseq(baseModel,math_cond, ranger_layer_factory)
model.summary()
'''