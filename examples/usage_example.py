from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from enum import Enum
import sys

LIBRARY_PATH = "../"

# appending a path
sys.path.append(LIBRARY_PATH) #CHANGE THIS LINE 

from custom_layers.ranger import *
from model_helper.ranger_model import *

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
    x_train = tf.expand_dims(x_train, axis=3, name=None)

    x_val = x_train[-2000:, :, :, :]
    y_val = y_train[-2000:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]

    return x_train, y_train, x_val, y_val

def build_model(input_shape):
    """
    Saved weights should be the path to a h5 file if you have already trained the model
    """
    
    inputs = keras.Input(shape=input_shape, name='input')
    conv1 = layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', name='conv1')(inputs)
    pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), name='maxpool1')(conv1)
    conv2 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same",
                          name='conv2')(pool1)
    pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), name='maxpool2')(conv2)
    conv3 = layers.Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same",
                          name='conv3')(pool2)
    
    flatten = layers.Flatten(name='flatten')(conv3)
    dense1 = layers.Dense(84, activation='relu', name='dense1')(flatten)
    outputs = layers.Dense(10, activation='softmax', name='dense3')(dense1)

    return keras.Model(inputs=(inputs,), outputs=outputs)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------EXAMPLE CODE------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Load Data from dataset
x_train, y_train, x_val, y_val = load_data()

#Build the model
model = build_model(x_train[0].shape)
model.summary()

#Load Model into Ranger Helper
RANGER = RANGER_HELPER(model)
#Add Ranger Layer after each Convolutions or Maxpool
RANGER.convert_model()
RANGER.get_model().summary()

exit()
#Disable Ranger Layers inside the model
RANGER.set_ranger_mode(RangerModes.Disabled)

#Train The network (Or Load the weights)

#Set Ranger Layers to Parameter Tuning mode
RANGER.set_ranger_mode(RangerModes.RangeTuning)

#Set Ranger Layers to Inference Mode
#Now we can test the effectiveness of the ranger approach