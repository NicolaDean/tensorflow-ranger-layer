import tensorflow as tf
from tensorflow import keras
from enum import Enum
import sys
from keras.engine import functional
from keras import Sequential
from tqdm import tqdm
from keras import losses

RANGER_MODULE_PATH = "../"
from .recursive_models import *

# appending a path
sys.path.append(RANGER_MODULE_PATH) #CHANGE THIS LINE

from custom_layers.ranger import *


class RANGER_HELPER():

    def __init__(self,model):
        self.model = model
        self.put_after = [keras.layers.Conv2D,keras.layers.MaxPool2D]

    def set_model(self,model):
        self.model = model
    

    def set_put_ranger_after(self,layers):
        self.put_after = layers

    '''
    Find all Ranger layer in the network and set theyr mode to a specific one
    '''
    def set_ranger_mode(self,mode=RangerModes.Inference,policy=RangerPolicies.Clipper,granularity=RangerGranularity.Layer):
        layers = [layer for layer in self.model.layers]
        for l in layers:
            if isinstance(l,Ranger):
                l.set_ranger_mode(mode)
                l.set_ranger_policy(policy)
                l.set_ranger_granularity(granularity)

    def reset_ranger_layers(self):
        layers = [layer for layer in self.model.layers]
        for l in layers:
            if isinstance(l,Ranger):
                l.reset_w()

    '''
    Recursively Explore Layer of model subblock in search of Conv and Maxpool to add Ranger after them
    '''
    def convert_block(layers,new_layer=keras.Sequential()) -> functional.Functional:

        for l in layers:
            if isinstance(l,keras.layers.Conv2D) or isinstance(l,keras.layers.MaxPool2D): #TODO PUT A FOR CYCLE ON "put_after"
                print(f"Added Ranger after layer: {l.name}")
                new_layer.add(l)
                new_layer.add(Ranger("ranger_" + l.name))
            elif isinstance(l,functional.Functional):
                block_layers = [layer for layer in l.layers]
                new_block = RANGER_HELPER.convert_block(block_layers,new_layer)
                #new_block = RANGER_HELPER.convert_block(block_layers)
                #new_layer.add(new_block)
                '''
                for x in new_block.layers:
                    new_layer.add(x)
                '''
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
        new_model = keras.Sequential()
        #IN and OUT of the Network
        #outputs = layers[len(layers)-1]
        #layers.pop()    #Remove output

        #Recursively Search every subblock to add Renger after Conv and Maxpool
        new_model = RANGER_HELPER.convert_block(layers,new_model)
        #new_model = keras.Model(inputs=(inputs), outputs=outputs)

        return new_model
    
    def convert_model_v2(self):
        def match_cond(layer):
            if     isinstance(layer,tf.keras.layers.Conv2D) \
                or isinstance(layer,tf.keras.layers.Add) \
                or isinstance(layer,tf.keras.layers.MaxPooling2D) \
                or isinstance(layer,tf.keras.layers.AveragePooling2D) \
                or isinstance(layer,tf.keras.layers.BatchNormalization):
                return True
            else:
                return False
        
        def ranger_layer_factory(layer):
            return Ranger(name=f"ranger")
        
        self.model = insert_layer_nonseq(self.model,match_cond, ranger_layer_factory)
    
    
    '''
    Convert model given in input to RANGER into a Ranger model (Add Ranger after each layer Conv2D or MaxPool)
    '''
    def convert_model(self):
        self.model = RANGER_HELPER.convert_model_from_src(self.model)

    '''
    Take in input a Dataset
    Compute inferences on the model and in the mean time it compute the range domain of all layers

    After it finish the process it set the model in inference mode
    '''
    def tune_model_range(self,X,Y=None):
        self.reset_ranger_layers()
        print("Tuning the moodel Range Domain")
        self.set_ranger_mode(RangerModes.RangeTuning)
        self.model.predict(X)

        #TODO => PRINT THE MODELS RANGES
        self.set_ranger_mode(RangerModes.Inference)

    '''
    Return the model generated by RANGER
    '''
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