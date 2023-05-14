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

    
Fault_injection_Report = make_dataclass("Fault_injection_Report", [("layer_name",str),("sample_id", int), ("num_of_injection", int), ("misclassifications", int),("experiment",str)])
Error_ID_report = make_dataclass("Error_ID_report", [("Mode", str), ("Layer_name", str), ("Sample_id", int), ("Cardinality", int), ("Pattern", int), ("Misprediction", int)])


CLASSES_MODULE_PATH = "/../"

# directory reach
directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory + CLASSES_MODULE_PATH)
from .recursive_models import *
from classes import *
import classes

original_models_path = str(pathlib.Path(classes.__file__).parent.absolute())


class CLASSES_MODELS_PATH:
    models      = original_models_path + "/models"
    models_warp = original_models_path + "/models_warp"


def gen_batch(x,y,batch_size):
    batch       = [x] * batch_size
    y_batch     = [y] * batch_size
    batch       = tf.stack(batch)
    y_batch     = tf.stack(y_batch)

    return batch,y_batch

def count_misclassification(predictions,labels, error_ids):
    labels      = tf.argmax(labels, 1)
    predictions = tf.reshape(predictions,labels.shape)

    labels      = tf.cast(labels        ,tf.int32)
    predictions = tf.cast(predictions   ,tf.int32)

    result      = tf.reduce_sum(tf.cast(tf.not_equal(labels,predictions), tf.int32)).numpy()
    
    misclassifications = tf.concat([tf.convert_to_tensor(error_ids, dtype = tf.int32), tf.expand_dims(tf.cast(tf.not_equal(labels, predictions), tf.int32), axis = 1)], axis = 1).numpy()

    return result, misclassifications

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
    
    def elaborate_layer(self,l,num_of_injection_sites):
            CLASSES_MODEL_TYPE = CLASSES_HELPER.check_classes_layer_compatibility(l)

            if CLASSES_MODEL_TYPE != None:
                
                print(f"Added Fault Layer after layer: {l.name} with Error Model [{CLASSES_MODEL_TYPE}]")
                
                shape = l.output_shape
                inverted_shape = (shape[0],shape[3],shape[1],shape[2])

                print(f"Shapes : ({shape} => {inverted_shape})")

                injection_layer_name = "classes_" + l.name

                self.injection_points.append(injection_layer_name)
                available_injection_sites, masks, error_ids = create_injection_sites_layer_simulator(num_of_injection_sites,
                                                                            CLASSES_MODEL_TYPE,
                                                                            str(inverted_shape), str(shape),CLASSES_MODELS_PATH.models_warp, return_id_errors = True)
                error_ids = np.array(error_ids)
                error_ids = np.squeeze(error_ids)
                return ErrorSimulator(available_injection_sites,masks,len(available_injection_sites),error_ids,name="classes")
            
    def convert_model_v2(self,num_of_injection_sites):
        
        def match_cond(layer):
            return CLASSES_HELPER.check_classes_layer_compatibility(layer) != None
        
        def classes_layer_factory(layer):
            return self.elaborate_layer(layer,num_of_injection_sites)
        
        self.vanilla_model = self.model
        self.num_of_injection = math.floor(num_of_injection_sites / 5)
        self.model = insert_layer_nonseq(self.model,match_cond, classes_layer_factory)

    '''
    Convert a Model by adding Injection points after each Convolution
    '''
    def convert_model(self,num_of_injection_sites):
        self.vanilla_model = self.model
        self.model = self.convert_model_from_src(self.model,num_of_injection_sites)
        self.num_of_injection = math.floor(num_of_injection_sites / 5)


    
    def get_model(self):
        return self.model
    
    '''
    Deprecated Do not work recursivly (TODO FIX)
    '''
    def set_mode(self, layer_name, mode: ErrorSimulatorMode):
        layer = self.model.get_layer("classes_" + layer_name)

        if isinstance(layer,ErrorSimulator):
            print(f"Changing mode to {layer.name}")
            layer.set_mode(mode)


    '''
    Return the most compatible error model for the input layer
    '''
    def check_classes_layer_compatibility(layer):
        #TODO => ADD ALL LAYERS
        if isinstance(layer,keras.layers.Add):
            return OperatorType['Add']#TODO FIX, ADD THE ADD MODEL TO WARP
        elif isinstance(layer,keras.layers.BatchNormalization):
            return OperatorType['FusedBatchNormV3']
        elif isinstance(layer,keras.layers.MaxPooling2D):
            return OperatorType['MaxPool2D']
        elif isinstance(layer,keras.layers.AveragePooling2D):
            return OperatorType['AvgPool2D']
        elif isinstance(layer,keras.layers.Conv2D):
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
                    #return OperatorType['Conv2D3x3S2']
                    return OperatorType['Conv2D3x3']
                else:
                    #return OperatorType['Conv2D3x3']
                    return OperatorType['Conv2D3x3']
            elif stride == 2:
                return OperatorType['Conv2D3x3']
                
        #TODO ADD OTHER CLASSES LAYER
        else:
            return None
    
    '''
    Take in input a classic CNN and add CLASSES ErrorSimulator layer for each Compatible Layer (All Layer for which we have an error Model)
    '''
    def convert_model_from_src(self,model: tf.keras.Model,num_of_injection_sites):
        layers      = [layer for layer in model.layers]
        new_model   = keras.Sequential()
        new_model   = self.convert_block(layers,num_of_injection_sites,new_model)

        return new_model

    '''
    Take in input a Model (or a generic Functional Layer) and give in output a Sequential layer with the added ErrorSimulator (Only for Classes compatible layers)
    '''
    def convert_block(self,layers,num_of_injection_sites,new_layer=keras.Sequential()) -> functional.Functional:

        for l in layers:
            CLASSES_MODEL_TYPE = CLASSES_HELPER.check_classes_layer_compatibility(l)

            if CLASSES_MODEL_TYPE != None:
                
                print(f"Added Fault Layer after layer: {l.name} with Error Model [{CLASSES_MODEL_TYPE}]")
                
                shape = l.output_shape
                inverted_shape = (shape[0],shape[3],shape[1],shape[2])

                print(f"Shapes : ({shape} => {inverted_shape})")

                injection_layer_name = "classes_" + l.name

                self.injection_points.append(injection_layer_name)
                available_injection_sites, masks, error_ids = create_injection_sites_layer_simulator(num_of_injection_sites,
                                                                            CLASSES_MODEL_TYPE,
                                                                            str(inverted_shape), str(shape),CLASSES_MODELS_PATH.models_warp, return_id_errors = True)
                error_ids = np.array(error_ids)
                error_ids = np.squeeze(error_ids)
                new_layer.add(l)
                new_layer.add(ErrorSimulator(available_injection_sites, masks,len(available_injection_sites), error_ids,name=injection_layer_name))

            elif isinstance(l,functional.Functional):
                block_layers = [layer for layer in l.layers]
                new_block = self.convert_block(block_layers,num_of_injection_sites,new_layer)
                #new_layer.add(new_block)
            else:
                new_layer.add(l)

        return new_layer

    #def convert_block_v2(self,layers,num_of_injection_sites,new_layer=keras.Sequential()) -> functional.Functional:

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

    def get_layer(model,layer_name):
        layers      = [layer for layer in model.layers]
        for layer in layers:
            if isinstance (layer, keras.Model):
                # Check if there is one sub_model
                sub_model = layer
                return CLASSES_HELPER.get_layer(sub_model,layer_name)
            elif layer.name == layer_name:
                return layer

        
    '''
    Disable All FaultInjection Layers
    '''
    def disable_all(self):
        print(self.injection_points)
        for l in self.injection_points:
            #print(f"Injection point {l}")
            layer = CLASSES_HELPER.get_layer(self.model,l)
            layer.set_mode(ErrorSimulatorMode.disabled)
    
    '''
    Given a Dataset and an injection point
    return the report of injection campaing on that layer
    '''
    def get_layer_injection_report(self,layer_name,X,Y,experiment_name="Generic",num_of_iteration=100):
        #TODO Check that this is an ErrorSimulator Layer

        self.disable_all()                          #Disable all InjectionPoints
        #layer = self.model.get_layer(layer_name)    #Get the selected Injection point
        layer = CLASSES_HELPER.get_layer(self.model,layer_name)
        layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print(f"----Fault Campaign with {layer_name} enabled-----")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        report = []
        ids_report = []
        vanilla_errors = 0
        #For each Sample in the dataset
        for i,dataset in enumerate(zip(X,Y)):
            x,y = dataset

            vanilla_res = self.vanilla_model.predict(np.expand_dims(x, 0), verbose = 0) #TODO make it using a "Vanilla-mask" instead of predicting every time
            #print(f"[{np.argmax(vanilla_res,axis=-1)}] => {y}")
            if np.argmax(vanilla_res,axis=-1) == y:
                BATCH_SIZE = 64
                x_batch,y_batch = gen_batch(x,y,batch_size=self.num_of_injection)
                self.model.run_eagerly=True
                pred = self.model.predict(x_batch,batch_size=BATCH_SIZE)
                ids = [layer.error_ids[i] for i in np.squeeze(layer.get_history()[1:])]
                
                errors, misclassifications = count_misclassification(y_batch,pred, ids)
                
                '''
                #For "Num_of_injection" run we inject a random fault using the error model
                errors = 0
                for _ in range(self.num_of_injection):
                    res = self.model.predict(np.expand_dims(x, 0), verbose = 0)
                    if np.argmax(res) != y:
                        errors += 1
                '''
                line_report = Fault_injection_Report(layer_name,i,self.num_of_injection,errors,experiment_name)
                report += [line_report]

                for injection in misclassifications:
                    ids_report += [Error_ID_report(experiment_name, layer_name, i, injection[0], injection[1], injection[2])]
                print(f'[Layer: {layer_name}] => [Sample: {i}] Number of misclassification over {self.num_of_injection} injections: {errors}') 
            else:
                print(f"Sample: {i} Misclassified by vanilla model")
                vanilla_errors += 1
            layer.clear_history()

        #report = pd.DataFrame(report)
        #print(report)
        return report, ids_report 
    '''
    Given a Dataset:
    For each injection point inside the model:
        Run an epoch
        Compute accuracy, Misclassification etc... [If exist, take mask in consideration, if not generate it automatically]
    Generate a detailed report in Dataframe format

    1) concat_previous  = True will concatenate the previously generated report with the new one to facilitate creation of dataframe for complex experiments
    2) fault_tollerance = True will simply add to the Dataframe a column to indicate the presence of FaultTollerance techniques
    '''
    def gen_model_injection_report(self,X,Y,experiment_name="Generic",num_of_iteration=100,concat_previous=False,file_name_report="report.csv",file_name_patterns="patterns.csv") -> pd.DataFrame:

        #For Each Injection Point
        report = []
        error_prof_report = []
        for l in self.injection_points:
            layer_report, ids_report = self.get_layer_injection_report(l,X,Y,experiment_name,num_of_iteration)
            report += layer_report
            error_prof_report += ids_report

        report = pd.DataFrame(report)
        error_prof_report = pd.DataFrame(error_prof_report)
        print(report)

        if concat_previous:
            mode = 'a'
            header=False
        else:
            mode = 'w'
            header=True

        print(f"Saving {file_name_report} with mode {mode}")
        report.to_csv(file_name_report,mode=mode,header=header)
        print(f"Saving {file_name_patterns} with mode {mode}")
        error_prof_report.to_csv(file_name_patterns,mode=mode,header=header)
        return report, error_prof_report
    
            

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