from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from enum import Enum
import sys
import os
import pathlib
from keras.utils import img_to_array, array_to_img
from tqdm import tqdm



WEIGHT_FILE_PATH = "../saved_models/"
LIBRARY_PATH = "/../"

# directory reach
directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + LIBRARY_PATH)

print("AAA:" + directory + LIBRARY_PATH)
from model_helper.ranger_model import *
from model_helper.classes_model import *
from datasets import gtsrb

VALIDATION_SIZE = 100
MODEL_NAME = "vgg_mnist"

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------EXAMPLE CODE------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
#--------------------------MODEL CREATION + DATASET LOADING----------------------------------------
#--------------------------------------------------------------------------------------------------

#Load Data from dataset
x_train,x_val,y_train,y_val = gtsrb.load_train()
x_test,y_test               = gtsrb.load_test()

model = tf.keras.models.load_model("../saved_models/vgg19_gtsrb")
model.summary()

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


NUM_INJECTIONS = 128

num_requested_injection_sites = NUM_INJECTIONS * 5
#Load Model into Ranger Helper
CLASSES = CLASSES_HELPER(ranger_model)         #PROBLEM HERE (??? TODO FIX ???) => With model work, with ranger_model not.. why??

#Add Fault Injection Layer after each Convolutions or Maxpool
CLASSES.convert_model_v2(num_requested_injection_sites)
classes_model = CLASSES.get_model()
#classes_model.predict(x_val)
classes_model.summary()

CLASSES.disable_all() #Disable all fault injection points

RANGER.set_model(classes_model) #IMPORTANT (otherwise Ranger.set_ranger_mode would not work!)

#--------------------------------------------------------------------------------------------------
#--------------------------FAULT CAMPAIGN + REPORT GENERATION--------------------------------------
#--------------------------------------------------------------------------------------------------


print("---------MODELS COMPARISON----------------")
x_val = x_val[:10]
y_val = y_val[:10]

y_val=np.argmax(y_val,axis=-1)
#TODO => USE THE TEST SET FOR NOT BIASED TESTING
#CLASSES.get_layer_injection_report("classes_conv2d_1",x_val,y_val)
RANGER.set_ranger_mode(RangerModes.Disabled)
vanilla = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "FaultInjection",concat_previous=True)

RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Layer)
clipping_layer  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Clipping_Layer",concat_previous=True)
RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Ranger,RangerGranularity.Layer)
ranger_layer  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Ranger_Layer",concat_previous=True)

RANGER.set_ranger_mode(granularity = RangerGranularity.Value)
RANGER.tune_model_range(x_train)

RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Value)
clipping_value  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Clipping_Value",concat_previous=True)
RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Ranger,RangerGranularity.Value)
ranger_value  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Ranger_Value",concat_previous=True)


#TODO ADD Clipping_Layer , Threshold_Value, Threshold_layer

report = pd.concat([vanilla,clipping_layer, ranger_layer, clipping_value, ranger_value])
report.to_csv("vgg19_gtsrb_report.csv")

print(report)



'''
batch       = [x_val[NUM]] * NUM_INJECTIONS
y_batch     = [y_val[NUM]] * NUM_INJECTIONS
batch       = tf.stack(batch)
y_batch     = tf.stack(y_batch)

print(y_batch.shape)
print(batch.shape)
'''