from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from enum import Enum
import sys
import os
import pathlib
from keras.utils import img_to_array, array_to_img
from tqdm import tqdm


REPORT_PATH = "../reports/"
WEIGHT_FILE_PATH = "../saved_models/"
LIBRARY_PATH = "/../"

# directory reach
directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + LIBRARY_PATH)

print("AAA:" + directory + LIBRARY_PATH)
from model_helper.ranger_model import *
from model_helper.classes_model import *

def add_classes_to_model(model,layer_name,NUM_INJECTIONS=128):
    num_requested_injection_sites = NUM_INJECTIONS * 5
    #Load Model into Ranger Helper
    CLASSES = CLASSES_HELPER(model)      

    #Add Fault Injection Layer after each Convolutions or Maxpool
    CLASSES.add_classes_by_name(layer_name,num_requested_injection_sites)
    classes_model = CLASSES.get_model()
    #classes_model.predict(x_val)
    classes_model.summary()
    #keras.utils.plot_model(classes_model,to_file="classes.png" ,show_shapes=True)
    CLASSES.disable_all() #Disable all fault injection points

    return CLASSES

def add_ranger_classes_to_model(model,layer_name,NUM_INJECTIONS=128):
    #--------------------------------------------------------------------------------------------------
    #--------------------------RANGER SETUP------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------

    #Load Model into Ranger Helper
    RANGER = RANGER_HELPER(model)

    #Add Ranger Layer after each Convolutions or Maxpool
    RANGER.convert_model_v2()
    #tf.executing_eagerly()

    #Extract the new Model containing Ranger
    ranger_model = RANGER.get_model()
    ranger_model.summary()
    
    #TUNE THE LAYERS RANGE DOMAIN
    #RANGER.tune_model_range(x_train)
    
    #--------------------------------------------------------------------------------------------------
    #--------------------------CLASSES SETUP-----------------------------------------------------------
    #--------------------------------------------------------------------------------------------------


    num_requested_injection_sites = NUM_INJECTIONS * 5
    #Load Model into Ranger Helper
    CLASSES = CLASSES_HELPER(ranger_model)        
    
    #Add Fault Injection Layer after each Convolutions or Maxpool
    CLASSES.add_classes_by_name(layer_name,num_requested_injection_sites)
    classes_model = CLASSES.get_model()
    #classes_model.predict(x_val)
    classes_model.summary()
    #keras.utils.plot_model(classes_model,to_file="classes.png" ,show_shapes=True)
    CLASSES.disable_all() #Disable all fault injection points

    RANGER.set_model(classes_model) #IMPORTANT (otherwise Ranger.set_ranger_mode would not work!)

    return RANGER,CLASSES

def run_ranger_experiment(model,x_train,x_val,y_train,y_val,experiment_name,NUM_INJECTIONS=128):
    #--------------------------------------------------------------------------------------------------
    #--------------------------RANGER SETUP------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------

    #Load Model into Ranger Helper
    RANGER = RANGER_HELPER(model)

    #Add Ranger Layer after each Convolutions or Maxpool
    RANGER.convert_model_v2()
    #tf.executing_eagerly()

    #Extract the new Model containing Ranger
    ranger_model = RANGER.get_model()
    ranger_model.summary()
    
    #TUNE THE LAYERS RANGE DOMAIN
    RANGER.tune_model_range(x_train)
    
    #--------------------------------------------------------------------------------------------------
    #--------------------------CLASSES SETUP-----------------------------------------------------------
    #--------------------------------------------------------------------------------------------------


    num_requested_injection_sites = NUM_INJECTIONS * 5
    #Load Model into Ranger Helper
    CLASSES = CLASSES_HELPER(ranger_model)         #PROBLEM HERE (??? TODO FIX ???) => With model work, with ranger_model not.. why??

    #Add Fault Injection Layer after each Convolutions or Maxpool
    CLASSES.convert_model(num_requested_injection_sites)
    classes_model = CLASSES.get_model()
    #classes_model.predict(x_val)
    classes_model.summary()
    CLASSES.disable_all() #Disable all fault injection points

    RANGER.set_model(classes_model) #IMPORTANT (otherwise Ranger.set_ranger_mode would not work!)

    #--------------------------------------------------------------------------------------------------
    #--------------------------FAULT CAMPAIGN + REPORT GENERATION--------------------------------------
    #--------------------------------------------------------------------------------------------------
    base_name    = experiment_name + ".csv"
    file_name    = REPORT_PATH + base_name
    file_pattern = REPORT_PATH + "pattern_"+ base_name
    filesummary  = REPORT_PATH + experiment_name + "_Summary.csv"

    print("---------MODELS COMPARISON----------------")

    #TODO => USE THE TEST SET FOR NOT BIASED TESTING
    #CLASSES.get_layer_injection_report("classes_conv2d_1",x_val,y_val)
    RANGER.set_ranger_mode(RangerModes.Disabled)
    report = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "FaultInjection",concat_previous=False,file_name_report=file_name,file_name_patterns=file_pattern,file_summary=filesummary)
    RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Layer)
    report  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Clipping_Layer",concat_previous=True,file_name_report=file_name,file_name_patterns=file_pattern,file_summary=filesummary)
    RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Ranger,RangerGranularity.Layer)
    report  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Ranger_Layer",concat_previous=True,file_name_report=file_name,file_name_patterns=file_pattern,file_summary=filesummary)

    '''
    RANGER.set_ranger_mode(granularity = RangerGranularity.Value)
    RANGER.tune_model_range(x_train)

    RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Value)
    report  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Clipping_Value",concat_previous=True,file_name_report=file_name,file_name_patterns=file_pattern)
    RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Ranger,RangerGranularity.Value)
    report  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Ranger_Value",concat_previous=True,file_name_report=file_name,file_name_patterns=file_pattern)
    '''