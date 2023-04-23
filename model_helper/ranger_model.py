import tensorflow as tf
from tensorflow import keras
from enum import Enum
import sys
from keras.engine import functional
from keras import Sequential
RANGER_MODULE_PATH = "../"

# appending a path
sys.path.append(RANGER_MODULE_PATH) #CHANGE THIS LINE

from custom_layers.ranger import *

class RANGER_HELPER():

    def __init__(self,model):
        self.model = model

    '''
    Find all Ranger layer in the network and set theyr mode to a specific one
    '''
    def set_ranger_mode(self,mode:RangerModes):
        layers = [layer for layer in self.model.layers]
        for l in layers:
            if isinstance(l,Ranger):
                l.set_ranger_mode(mode)

    '''
    Recursively Explore Layer of model subblock in search of Conv and Maxpool to add Ranger after them
    '''
    def convert_block(layers) -> functional.Functional:
        new_layer = keras.Sequential()

        for l in layers:
            if isinstance(l,functional.Functional):
                block_layers = [layer for layer in l.layers]
                new_block = RANGER_HELPER.convert_block(block_layers)
                new_layer.add(new_block)
            if isinstance(l,keras.layers.Conv2D) or isinstance(l,keras.layers.MaxPool2D):
                print(f"Added Ranger after layer: {l.name}")
                new_layer.add(l)
                new_layer.add(Ranger("ranger_" + l.name))
            else:
                new_layer.add(l)
        return new_layer
        
    '''
    Take in input a classic CNN and add ranger layer for each Convolution
    TODO MAKE THIS PROCESS RECURSIVE SO THAT WE CAN CONVERT ALSO COMPEX MODEL LIKE RESNET (or models that use subblocks of layers)
    '''
    def convert_model_from_src(model: tf.keras.Model):
        #TODO NOT IMPLEMENTED YET
        layers = [layer for layer in model.layers]

        #IN and OUT of the Network
        #outputs = layers[len(layers)-1]
        #layers.pop()    #Remove output

        #Recursively Search every subblock to add Renger after Conv and Maxpool
        new_model = RANGER_HELPER.convert_block(layers)
        #new_model = keras.Model(inputs=(inputs), outputs=outputs)

        return new_model
    
    def convert_model(self):
        self.model = RANGER_HELPER.convert_model_from_src(self.model)

    def get_model(self):
        return self.model



'''
IDEAS ON HOW IT SHOULD WORK

input = InputLayer(...)

x = Conv2D(...)(input)
x = Conv2D(...)(x)
....
x = Conv2D(...)(x)
output = Dense(...)(x)

model = keras.Model(input, output, "My_ranger_Model")
model.compile(...)

#Load model into ranger
RANGER = RANGER_HELPER(model)
RANGER.set_ranger_mode(RangerModes.Training)

[... TRAIN THE MODEL ...]

RANGER.set_ranger_mode(RangerModes.RangeTuning)

[... Run one epochs  to extract domain...]

RANGER.set_ranger_mode(RangerModes.Inference)

[... Now we can integrate Classes layers and test Ranger stuff ...]

'''