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
MODEL_NAME = "Lenet3"

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
    x_train = tf.expand_dims(x_train, axis=3, name=None)

    x_val = x_train[-VALIDATION_SIZE:, :, :, :]
    y_val = y_train[-VALIDATION_SIZE:]
    #x_train = x_train[:-2000, :, :, :]
    #y_train = y_train[:-2000]
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
        model = LeNet(x_train[0].shape)
        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
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

NUM_INJECTIONS = 128
NUM = 42
num_requested_injection_sites = NUM_INJECTIONS * 5


batch       = [x_val[NUM]] * NUM_INJECTIONS
y_batch     = [y_val[NUM]] * NUM_INJECTIONS
batch       = tf.stack(batch)
y_batch     = tf.stack(y_batch)


#--------------------------------------------------------------------------------------------------
#--------------------------RANGER SETUP------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Load Model into Ranger Helper
RANGER = RANGER_HELPER(model)

#Add Ranger Layer after each Convolutions or Maxpool
RANGER.convert_model()
RANGER.get_model().summary()
#tf.executing_eagerly()
#Extract the new Model containing Ranger
ranger_model = RANGER.get_model()
ranger_model.summary()

#TUNE THE LAYERS RANGE DOMAIN
RANGER.tune_model_range(x_val)




#accuracy = accuracy_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1))
#--------------------------------------------------------------------------------------------------
#--------------------------CLASSES SETUP-----------------------------------------------------------
#--------------------------------------------------------------------------------------------------


#Load Model into Ranger Helper
CLASSES = CLASSES_HELPER(model)

#Add Ranger Layer after each Convolutions or Maxpool
CLASSES.convert_model(num_requested_injection_sites)
classes_model = CLASSES.get_model()
classes_model.summary()
#classes_model.run_eagerly = True

CLASSES.set_mode("conv2d",ErrorSimulatorMode.disabled)
CLASSES.set_mode("conv2d_1",ErrorSimulatorMode.enabled)
CLASSES.set_mode("conv2d_2",ErrorSimulatorMode.disabled)


model           .compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
ranger_model    .compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
classes_model   .compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

'''
model           .evaluate(batch, y_batch, batch_size=128)
ranger_model    .evaluate(batch, y_batch, batch_size=128)
classes_model   .evaluate(batch, y_batch, batch_size=128)
'''

vanilla_p          = model         .predict(batch,batch_size=64)
classes_p          = classes_model .predict(batch,batch_size=64)
classes_p2         = classes_model .predict(batch,batch_size=64)
classes_p3         = classes_model .predict(batch,batch_size=64)


print(y_batch.shape)
vanilla_misc = count_misclassification(y_batch,vanilla_p)
classes_misc = count_misclassification(y_batch,classes_p)
classes_p2   = count_misclassification(y_batch,classes_p2)
classes_p3   = count_misclassification(y_batch,classes_p3)
print(f"Vanilla [ {vanilla_misc} ] vs Classes [ {classes_misc} ]")

#RANGER.set_model(classes_model)
