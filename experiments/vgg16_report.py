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
from models import LeNet
from models import VGG16

VALIDATION_SIZE = 100
MODEL_NAME = "vgg_mnist"


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = np.dstack([x_train] * 3)
    x_test = np.dstack([x_test] * 3)

    x_train = x_train.reshape(-1, 28,28,3)
    x_test = x_test.reshape(-1,28,28,3)

    x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
    x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])

    x_val = x_train[-VALIDATION_SIZE:, :, :, :]
    y_val = y_train[-VALIDATION_SIZE:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]

    return x_train, y_train, x_val, y_val

def build_model(load_model_from_memory=False):
    #Build the model
    path_weights = os.path.join(WEIGHT_FILE_PATH,MODEL_NAME)
    print(f"Load weights from => {path_weights}")
    if path_weights is not None and load_model_from_memory:
        model = keras.models.load_model(path_weights)
    else:
        print(f"NO MODEL FAULD AT {path_weights} => Loading Classic LeNet")
        model = VGG16(x_train[0].shape)
        
        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_val, y_val))
        model.save(WEIGHT_FILE_PATH + MODEL_NAME)
    model.summary()
    return model


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------EXAMPLE CODE------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
#--------------------------MODEL CREATION + DATASET LOADING----------------------------------------
#--------------------------------------------------------------------------------------------------

#Load Data from dataset
x_train, y_train, x_val, y_val = load_data()

LOAD_MODEL = True
model = build_model(LOAD_MODEL)

#--------------------------------------------------------------------------------------------------
#--------------------------RANGER SETUP------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Load Model into Ranger Helper
RANGER = RANGER_HELPER(model)

#Add Ranger Layer after each Convolutions or Maxpool
RANGER.convert_model()
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
CLASSES.convert_model(num_requested_injection_sites)
classes_model = CLASSES.get_model()
classes_model.predict(x_val)
classes_model.summary()

CLASSES.disable_all() #Disable all fault injection points

RANGER.set_model(classes_model) #IMPORTANT (otherwise Ranger.set_ranger_mode would not work!)

#--------------------------------------------------------------------------------------------------
#--------------------------FAULT CAMPAIGN + REPORT GENERATION--------------------------------------
#--------------------------------------------------------------------------------------------------


print("---------MODELS COMPARISON----------------")

#TODO => USE THE TEST SET FOR NOT BIASED TESTING
#CLASSES.get_layer_injection_report("classes_conv2d_1",x_val,y_val)
RANGER.set_ranger_mode(RangerModes.Disabled)
vanilla, vanilla_ids = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "FaultInjection",concat_previous=True)

RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Layer)
clipping_layer, clipping_layer_ids  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Clipping_Layer",concat_previous=True)
RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Ranger,RangerGranularity.Layer)
ranger_layer, ranger_layer_ids  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Ranger_Layer",concat_previous=True)

RANGER.set_ranger_mode(granularity = RangerGranularity.Value)
RANGER.tune_model_range(x_train)

RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Value)
clipping_value, clipping_value_ids  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Clipping_Value",concat_previous=True)
RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Ranger,RangerGranularity.Value)
ranger_value, ranger_value_ids  = CLASSES.gen_model_injection_report(x_val,y_val,experiment_name = "Ranger_Ranger_Value",concat_previous=True)


#TODO ADD Clipping_Layer , Threshold_Value, Threshold_layer

report = pd.concat([vanilla,clipping_layer, ranger_layer, clipping_value, ranger_value])
report.to_csv("vgg16_ranger_mnist.csv")
print(report)


report = pd.concat([vanilla_ids, clipping_layer_ids, ranger_layer_ids, clipping_value_ids, ranger_value_ids])
report.to_csv("vgg16_ranger_mnist_mispredictions.csv")





'''
batch       = [x_val[NUM]] * NUM_INJECTIONS
y_batch     = [y_val[NUM]] * NUM_INJECTIONS
batch       = tf.stack(batch)
y_batch     = tf.stack(y_batch)

print(y_batch.shape)
print(batch.shape)
'''