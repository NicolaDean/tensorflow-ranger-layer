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


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = np.dstack([x_train] * 3)
    x_test = np.dstack([x_test] * 3)

    x_train = x_train.reshape(-1, 28,28,3)
    x_test = x_test.reshape(-1,28,28,3)

    x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
    x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])

    x_val = x_train[-100:, :, :, :]
    y_val = y_train[-100:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]

    return x_train, y_train, x_val, y_val

def build_model(load_model_from_memory=False):
    #Build the model
    path_weights = os.path.join(WEIGHT_FILE_PATH,'VGG16')
    print(f"Load weights from => {path_weights}")
    if path_weights is not None and load_model_from_memory:
        model = keras.models.load_model(path_weights)
    else:
        print(f"NO MODEL FAULD AT {path_weights} => Loading Classic LeNet")
        model = LeNet(x_train[0].shape)
        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=2)

        model.save(WEIGHT_FILE_PATH + "Vgg16")
    model.summary()
    return model
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------EXAMPLE CODE------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Load Data from dataset
x_train, y_train, x_val, y_val = load_data()

LOAD_MODEL = True
model = build_model(LOAD_MODEL)

NUM_INJECTIONS = 100
NUM = 42

num_requested_injection_sites = NUM_INJECTIONS * 5

#Load Model into Ranger Helper
CLASSES = CLASSES_HELPER(model)

#Add Ranger Layer after each Convolutions or Maxpool
CLASSES.convert_model(num_requested_injection_sites)
classes_model = CLASSES.get_model()
classes_model.summary()
classes_model.run_eagerly = True

CLASSES.set_mode("conv2d",ErrorSimulatorMode.disabled)
CLASSES.set_mode("conv2d_1",ErrorSimulatorMode.disabled)
CLASSES.set_mode("conv2d_2",ErrorSimulatorMode.enabled)

print("---------MODELS COMPARISON----------------")
#CLASSES.get_layer_injection_report("classes_conv2d_1",x_val,y_val)
report = CLASSES.gen_model_injection_report(x_val,y_val,concat_previous=True)

report.to_csv("report_" + classes_model.name)
