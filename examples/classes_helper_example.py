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
    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
    x_train = tf.expand_dims(x_train, axis=3, name=None)

    x_val = x_train[-200:, :, :, :]
    y_val = y_train[-200:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]

    return x_train, y_train, x_val, y_val

def load_data_vgg():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = np.dstack([x_train] * 3)
    x_test = np.dstack([x_test] * 3)

    x_train = x_train.reshape(-1, 28,28,3)
    x_test = x_test.reshape(-1,28,28,3)

    x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
    x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])

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

LOAD_MODEL = True

#Build the model
path_weights = os.path.join(WEIGHT_FILE_PATH,'Lenet2')
print(f"Load weights from => {path_weights}")
if path_weights is not None and LOAD_MODEL:
    model = keras.models.load_model(path_weights)
else:
    print(f"NO MODEL FAULD AT {path_weights} => Loading Classic LeNet")
    model = LeNet(x_train[0].shape)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=2)

    model.save(WEIGHT_FILE_PATH + "Lenet2")
model.summary()

def test_sample(model,sample_index,name):
    errors = 0
    for _ in tqdm(range(NUM_INJECTIONS)):
        res = model.predict(np.expand_dims(x_val[sample_index], 0), verbose = 0)
        if np.argmax(res) != y_val[sample_index]:
            errors += 1
    print(f'[{name}] Number of misclassification over {NUM_INJECTIONS} injections: {errors}')


NUM_INJECTIONS = 100
NUM = 42

num_requested_injection_sites = NUM_INJECTIONS * 5



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

#Load Model into Ranger Helper
CLASSES = CLASSES_HELPER(ranger_model)
#Add Ranger Layer after each Convolutions or Maxpool
CLASSES.convert_model(num_requested_injection_sites)
classes_model = CLASSES.get_model()
classes_model.summary()
classes_model.run_eagerly = True

CLASSES.set_mode("conv2d",ErrorSimulatorMode.disabled)
CLASSES.set_mode("conv2d_1",ErrorSimulatorMode.enabled)
CLASSES.set_mode("conv2d_2",ErrorSimulatorMode.disabled)

RANGER.set_model(classes_model)

print("---------MODELS COMPARISON----------------")
test_sample(model,NUM,"VANILLA")
test_sample(ranger_model,NUM,"RANGER")
RANGER.set_ranger_mode(RangerModes.Disabled)
test_sample(classes_model,NUM,"CLASSES")
RANGER.set_ranger_mode(RangerModes.Inference)
test_sample(classes_model,NUM,"CLASSES + RANGER-CLIPPING")
exit()

#--------------Try Launching one fault per layer----------------------

layers_to_fault = ["conv1","conv2","conv3"]
layers_to_fault = ["conv2d","conv2d_1","conv2d_2","conv2d_3","conv2d_4","conv2d_5","conv2d_6","conv2d_6","conv2d_7","conv2d_8","conv2d_9","conv2d_9","conv2d_10","conv2d_11","conv2d_12"]

i = 0
for l in layers_to_fault:
    CLASSES.set_mode(l,ErrorSimulatorMode.enabled)
    classes_model = CLASSES.get_model()
    classes_model.predict(np.expand_dims(x_val[i], 0))
    i = i + 1

exit()
