from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from keras.engine import functional
from keras import Sequential
import sys
import os
import pathlib
from tqdm import tqdm
import math

import pandas as pd
from dataclasses import make_dataclass

    
Fault_injection_Report = make_dataclass("Fault_injection_Report", [("layer_name",str),("sample_id", int), ("num_of_injection", int), ("misclassifications", int),("fault_tollerance",bool)])


CLASSES_MODULE_PATH = "/../"

# directory reach
directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory + CLASSES_MODULE_PATH)

from classes import *
import classes

original_models_path = str(pathlib.Path(classes.__file__).parent.absolute())


class CLASSES_MODELS_PATH:
    models      = original_models_path + "/models"
    models_warp = original_models_path + "/models_warp"



'''
This is an HELPER class to easily work with the CLASSES framwork on existing models (without having to write boilerplate code by hand)
'''
class CLASSES_HELPER():
    def __init__(self,model:tf.keras.Model) -> OperatorType:
        self.model              = model #Model that we want to test.
        self.injection_points   = []
        self.misc_mask          = None  #Contain a mask with 0 if vanilla model classify wrong, 1 if vanilla model classify ok

    def set_model(self,model):
        self.model = model
    def convert_model(self,num_of_injection_sites):
        self.model = self.convert_model_from_src(self.model,num_of_injection_sites)
        self.num_of_injection = math.floor(num_of_injection_sites / 5)

    
    def get_model(self):
        return self.model
    
    def set_mode(self, layer_name, mode: ErrorSimulatorMode):
        layer = self.model.get_layer("classes_" + layer_name)

        if isinstance(layer,ErrorSimulator):
            print(f"Changing mode to {layer.name}")
            layer.set_mode(mode)


    '''
    Return the most compatible error model for the input layer
    '''
    def check_classes_layer_compatibility(layer):
        if isinstance(layer,keras.layers.Conv2D):
            stride = layer.strides[0]
            kernel = layer.kernel_size[0]
            print(f"Stride = ({stride} , Kernel ({kernel}))")
            if stride == 0:
                return OperatorType['Conv2D3x3']
            elif stride == 1:
                if kernel == 1:
                    return OperatorType.Conv2D1x1
                elif kernel == 3:
                    #return OperatorType.Conv2D
                    return OperatorType['Conv2D3x3S2']
                else:
                    #return OperatorType['Conv2D3x3']
                    return OperatorType['Conv2D']
            elif stride == 2:
                if kernel == 2:
                    return OperatorType.Conv2D3x3S2
                else:
                    return None
                
        #TODO ADD OTHER CLASSES LAYER
        else:
            return None
    
    '''
    Take in input a classic CNN and add CLASSES ErrorSimulator layer for each Compatible Layer (All Layer for which we have an error Model)
    '''
    def convert_model_from_src(self,model: tf.keras.Model,num_of_injection_sites):
        layers      = [layer for layer in model.layers]
        new_model   = self.convert_block(layers,num_of_injection_sites)

        return new_model

    '''
    Take in input a Model (or a generic Functional Layer) and give in output a Sequential layer with the added ErrorSimulator (Only for Classes compatible layers)
    '''
    def convert_block(self,layers,num_of_injection_sites) -> functional.Functional:
        new_layer = keras.Sequential()
        
        for l in layers:
            CLASSES_MODEL_TYPE = CLASSES_HELPER.check_classes_layer_compatibility(l)

            if CLASSES_MODEL_TYPE != None:
                
                print(f"Added Fault Layer after layer: {l.name} with Error Model [{CLASSES_MODEL_TYPE}]")
                
                shape = l.output_shape
                inverted_shape = (shape[0],shape[3],shape[1],shape[2])

                print(f"Shapes : ({shape} => {inverted_shape})")

                injection_layer_name = "classes_" + l.name

                self.injection_points.append(injection_layer_name)
                available_injection_sites, masks = create_injection_sites_layer_simulator(num_of_injection_sites,
                                                                            CLASSES_MODEL_TYPE,
                                                                            str(inverted_shape), str(shape),CLASSES_MODELS_PATH.models)
                
                new_layer.add(l)
                new_layer.add(ErrorSimulator(available_injection_sites, masks,len(available_injection_sites),name=injection_layer_name))

            elif isinstance(l,functional.Functional):
                block_layers = [layer for layer in l.layers]
                new_block = self.convert_block(block_layers)
                new_layer.add(new_block)
            else:
                new_layer.add(l)

        return new_layer

    def get_included_models_path():
        return CLASSES_MODELS_PATH.models
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
    Return name of all injection point added to this model
    '''
    def get_injection_points(self):
        return self.injection_points

    '''
    Disable All FaultInjection Layers
    '''
    def disable_all(self):
        for l in self.injection_points:
            layer = self.model.get_layer(l)
            layer.set_mode(ErrorSimulatorMode.disabled)
    
    '''
    Given a Dataset and an injection point
    return the report of injection campaing on that layer
    '''
    def get_layer_injection_report(self,layer_name,X,Y,is_fault_tollerance_on=False):
        #TODO Check that this is an ErrorSimulator Layer

        self.disable_all()                          #Disable all InjectionPoints
        layer = self.model.get_layer(layer_name)    #Get the selected Injection point
        layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print(f"----Fault Campaign with {layer_name} enabled-----")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        report = []
        #For each Sample in the dataset
        for i,dataset in enumerate(zip(X,Y)):
            x,y = dataset
            #For "Num_of_injection" run we inject a random fault using the error model
            errors = 0
            for _ in range(self.num_of_injection):
                res = self.model.predict(np.expand_dims(x, 0), verbose = 0)
                if np.argmax(res) != y:
                    errors += 1
            line_report = Fault_injection_Report(layer_name,i,self.num_of_injection,errors,is_fault_tollerance_on)
            report += [line_report]
            print(f'[Layer: {layer_name}] => [Sample: {i}] Number of misclassification over {self.num_of_injection} injections: {errors}') 

        #report = pd.DataFrame(report)
        #print(report)
        return report
    '''
    Given a Dataset:
    For each injection point inside the model:
        Run an epoch
        Compute accuracy, Misclassification etc... [If exist, take mask in consideration, if not generate it automatically]
    Generate a detailed report in Dataframe format

    1) concat_previous  = True will concatenate the previously generated report with the new one to facilitate creation of dataframe for complex experiments
    2) fault_tollerance = True will simply add to the Dataframe a column to indicate the presence of FaultTollerance techniques
    '''
    def gen_model_injection_report(self,X,Y,fault_tollerance=False,concat_previous=False) -> pd.DataFrame:

        #For Each Injection Point
        report = []
        for l in self.injection_points:
            layer_report = self.get_layer_injection_report(l,X,Y)
            report += layer_report

        report = pd.DataFrame(report)
        print(report)
        return report
    
            

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