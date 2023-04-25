from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from keras.engine import functional
from keras import Sequential
import sys
import os
import pathlib

CLASSES_MODULE_PATH = "/../"

# directory reach
directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory + CLASSES_MODULE_PATH)

from classes import *

'''
Take in input a classic CNN and add CLASSES ErrorSimulator layer for each Compatible Layer (All Layer for which we have an error Model)
'''
def convert_model_from_src(model: tf.keras.Model,num_of_injection_sites):
    layers      = [layer for layer in model.layers]
    new_model   = convert_block(layers,num_of_injection_sites)

    return new_model

'''
Take in input a Model (or a generic Functional Layer) and give in output a Sequential layer with the added ErrorSimulator (Only for Classes compatible layers)
'''
def convert_block(layers,num_of_injection_sites) -> functional.Functional:
    new_layer = keras.Sequential()
    
    for l in layers:
        CLASSES_MODEL_TYPE = CLASSES_HELPER.check_classes_layer_compatibility(l)

        if CLASSES_MODEL_TYPE != None:
            print(f"Added Fault Layer after layer: {l.name}")

            shape = l.output_shape
            inverted_shape = (shape[0],shape[3],shape[1],shape[2])

            print(f"Shapes : ({shape} => {inverted_shape})")
            available_injection_sites, masks = create_injection_sites_layer_simulator(num_of_injection_sites,
                                                                          CLASSES_MODEL_TYPE,
                                                                          str(inverted_shape), str(shape))
            
            new_layer.add(l)
            new_layer.add(ErrorSimulator(available_injection_sites, masks, len(available_injection_sites),name="classes_" + l.name))

        elif isinstance(l,functional.Functional):
            block_layers = [layer for layer in l.layers]
            new_block = convert_block(block_layers)
            new_layer.add(new_block)
        else:
            new_layer.add(l)

    return new_layer


'''
This is an HELPER class to easily work with the CLASSES framwork on existing models (without having to write boilerplate code by hand)
'''
class CLASSES_HELPER():
    def __init__(self,model:tf.keras.Model) -> OperatorType:
        self.model      = model #Model that we want to test.
        self.misc_mask  = None  #Contain a mask with 0 if vanilla model classify wrong, 1 if vanilla model classify ok

    def convert_model(self,num_of_injection_sites):
        self.model = convert_model_from_src(self.model,num_of_injection_sites)

    
    def get_model(self):
        return self.model
    
    
    def check_classes_layer_compatibility(layer):
        if isinstance(layer,keras.layers.Conv2D):
            stride = layer.strides[0]
            kernel = layer.kernel_size[0]
            print(f"Stride = ({stride} , Kernel ({kernel}))")
            if stride == 1:
                if kernel == 1:
                    return OperatorType.Conv2D1x1
                elif kernel == 3:
                    return OperatorType.Conv2D3x3
                else:
                    print("MODEL USED IS CONV")
                    return OperatorType.Conv2D
            elif stride == 2:
                if kernel == 2:
                    return OperatorType.Conv2D3x3S2
                else:
                    return None
                
        #TODO ADD OTHER CLASSES LAYER
        else:
            return None

    '''
    Add an Injection point after layer with specified name
    '''
    def add_injections_from_names(self,layer_names,num_requested_injection_sites):

        target_layer = self.model.get_layer(layer_names)

        #Get layer index

        #Put idx-1 as input to Error Layer

        #Put Error Layer as input of layer.output

        return
    
    '''
    Create a boolean vector with following propertys:
        V[i] = 0 <=> i-th image is misclassified by vanilla model
        V[i] = 1 <=> i-th image is correctly classified by vanilla model

    INPUTS: Dataset and Label
    '''
    def extract_misclassification_mask(self,X,Y):
        return

    '''
    Given a Dataset:
    For each injection point inside the model:
        Run an epoch
        Compute accuracy, Misclassification etc... [If exist, take mask in consideration, if not generate it automatically]
    Generate a detailed report in Dataframe format
    '''
    def gen_injection_report(self,X,Y):
        return


'''
IDEAS ON HOW IT SHOULD WORK

#------------MODEL DEFINITION---------------------------------------
input = InputLayer(...)

x = Conv2D(...)(input)
x = Conv2D(...)(x)
....
x = Conv2D(...)(x)
output = Dense(...)(x)

#-----COMPILE THE MODEL AND TRAIN (OR LOAD EXISTING MODEL---------
model = RangerModel(input, output, "My_ranger_Model")
model.compile(...)

#------------ADD INJECTION POINTS---------------------------------

CLASSES = CLASSES_HELPER(model)                #Load the Model into Classes helper

CLASSES.extract_misclassification_mask(X,Y)    #Extract a mask to cut out from evaluation the vanilla misclassification.

CLASSES.add_injections_from_names("conv1",[other args])
CLASSES.add_injections_from_names("conv2",[other args])
CLASSES.add_injections_from_names("conv3",[other args])
CLASSES.add_injections_from_names("maxpool1",[other args])


#------------GENERATE REPORT---------------------------------
report = CLASSES.gen_injection_report(X,Y)


#NOW WE HAVE A REPORT IN DATAFRAME FORMAT
'''