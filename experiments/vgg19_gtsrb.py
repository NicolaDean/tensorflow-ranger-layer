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
from model_helper.run_experiment import *
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

x_val = x_test[:10]
y_val = y_test[:10]

y_val=np.argmax(y_val,axis=-1)

run_ranger_experiment(model,x_train,x_val,y_train,y_val,"vgg19_gtsrb_old_models",NUM_INJECTIONS=128)

