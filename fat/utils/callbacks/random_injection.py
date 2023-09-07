import tensorflow as tf
import keras
import copy
import sys
import random
import numpy as np
sys.path.append("./../../../model_helper/")

from model_helper.classes_model import *


'''
Activate Classes injection within a certain probability/frequency
'''
def layer_activation(self):
    active = tf.random.uniform([]) < self.extraction_frequency
    if active:
            #print("Active")
            #Disable previously selected injection point:
            layer = CLASSES_HELPER.get_layer(self.model,self.layer_name,verbose=False)
            layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
    else:
            #Disable previously selected injection point:
            #print("NOT active")
            layer = CLASSES_HELPER.get_layer(self.model,self.layer_name,verbose=False)
            layer.set_mode(ErrorSimulatorMode.disabled)  #Enable the Selected Injection point

class ClassesSingleLayerInjection(keras.callbacks.Callback):

    def __init__(self, CLASSES,layer_name,extraction_frequency = 1.0, use_batch = False):
        super().__init__()
        self.injection_points     = CLASSES.get_injection_points()
        self.CLASSES              = CLASSES
        self.extraction_frequency = extraction_frequency
        self.use_batch            = use_batch
        self.layer_name           = self.injection_points[0]

    def on_train_batch_begin(self, epoch, logs=None):
        if self.use_batch:
            layer_activation(self)

    def on_epoch_begin(self, epoch, logs=None):
        if not self.use_batch:
            layer_activation(self)
