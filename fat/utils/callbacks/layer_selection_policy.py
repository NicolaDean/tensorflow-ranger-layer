import tensorflow as tf
import keras

import sys
sys.path.append("./../../../model_helper/")

from model_helper.classes_model import *

class ClassesLayerPolicy(keras.callbacks.Callback):

    def __init__(self, CLASSES):
        super().__init__()
        self.injection_points = CLASSES.get_injection_points()
        self.CLASSES = CLASSES
    
    def on_epoch_begin(self, epoch, logs=None):
        #Select a random Injection point from the list
        selected_injection = tf.random.uniform(shape=(), minval=0, maxval=len(self.injection_points),dtype=tf.int32)
        #print(selected_injection)
        selected_injection = self.injection_points[selected_injection]

        #Disable all classes layers
        self.CLASSES.disable_all(verbose=False)

        #Enable the selected injection point:
        layer = CLASSES_HELPER.get_layer(self.model,selected_injection,verbose=False)
        layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point

        #Print the Name of the activated Layer
        keys = list(logs.keys())
        print("Injection point [ {} ] Activated for epoch: {};".format(selected_injection, epoch))