import tensorflow as tf
import keras

import sys
sys.path.append("./../../../model_helper/")

from model_helper.classes_model import *

class ClassesLayerPolicy(keras.callbacks.Callback):

    def __init__(self, injection_points, CLASSES):
        super().__init__()
        self.injection_points = injection_points
        self.CLASSES = CLASSES

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        #print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        #print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        #Select a random Injection point from the list
        selected_injection = tf.random.uniform(shape=[], minval=0., maxval=len(self.injection_points))
        selected_injection = self.injection_points[selected_injection]

        #Disable all classes layers
        self.CLASSES.disable_all()

        #Enable the selected injection point:
        layer = self.CLASSES.get_layer(self.model,selected_injection)
        layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point

        #Print the Name of the activated Layer
        keys = list(logs.keys())
        print("Start epoch {} of training with injection point [ {} ]; got log keys: {}".format(epoch, selected_injection, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        #print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        #print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        #print("...Training: end of batch {}; got log keys: {}".format(batch, keys))