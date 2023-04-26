from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from enum import Enum
import sys
import os

LIBRARY_PATH = "../"
WEIGHT_FILE_PATH = "../saved_models/"

# appending a path
sys.path.append(LIBRARY_PATH) #CHANGE THIS LINE 

from custom_layers.ranger import *
from model_helper.ranger_model import *
from model_helper.classes_model import *
#from models import VGG16
from models import LeNet

print(f" Tensorflow verion: {tf.__version__}")

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
    x_train = tf.expand_dims(x_train, axis=3, name=None)

    x_val = x_train[-2000:, :, :, :]
    y_val = y_train[-2000:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]

    return x_train, y_train, x_val, y_val

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------EXAMPLE CODE------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Load Data from dataset
x_train, y_train, x_val, y_val = load_data()

#Build the model
model = LeNet(x_train[0].shape)


path_weights = os.path.join(WEIGHT_FILE_PATH,'LeNet.h5')
print(f"Load weights from => {path_weights}")
if path_weights is not None:
    model = keras.models.load_model(path_weights)
else:
    print(f"NO MODEL FAULD AT {path_weights} => Loading Classic LeNet")
    model = LeNet(x_train[0].shape)

model.summary()
#Load Model into Ranger Helper
RANGER = RANGER_HELPER(model)
#Add Ranger Layer after each Convolutions or Maxpool
RANGER.convert_model()
RANGER.get_model().summary()

#Extract the new Model containing Ranger
ranger_model = RANGER.get_model()

#Compute the Range Domain of each Layer by putting Ranger in "RangeTuning" mode
RANGER.set_ranger_mode(RangerModes.RangeTuning)
#Emulate the tuning process
ranger_model.predict(np.expand_dims(x_val[4], 0)) 

#Now we can use Model with Clipped or Ranged values in inference mode
RANGER.set_ranger_mode(RangerModes.Inference)
ranger_model.predict(np.expand_dims(x_val[4], 0)) 

exit()
#Train The network (Or Load the weights)

CLASSES = CLASSES_HELPER(ranger_model)
#Add Ranger Layer after each Convolutions or Maxpool
CLASSES.convert_model(num_requested_injection_sites)
classes_model = CLASSES.get_model()
classes_model.summary()

