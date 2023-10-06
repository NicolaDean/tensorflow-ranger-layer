import tensorflow as tf
import keras
import copy
import sys
import random
import numpy as np
sys.path.append("./../../../model_helper/")

from model_helper.classes_model import *

from .random_injection import layer_activation


def selection_policy(self):
    #Disable previously selected injection point:
    layer = CLASSES_HELPER.get_layer(self.model,self.previous_injection,verbose=False)
    layer.set_mode(ErrorSimulatorMode.disabled)  #Enable the Selected Injection point

    #IF WE CHOSED TO UNIFORMLY ASSIGN INJECTION POINTS USE THE EXTRACTION STACK
    if self.uniform_extraction:
        #Shuffle the stack
        random.shuffle(self.current_stack)
        #Extract the selected layer
        self.layer_name         = self.current_stack.pop()
        
        if not self.current_stack:
            #Regenerate the stack when its empty
            self.current_stack = copy.deepcopy(self.injection_points)

    #IF WE CHOSED TO PICK AT RANDOM SIMPLY EXTRACT THE INDEX
    else:
        #Select a random Injection point from the list
        selected_injection = tf.random.uniform(shape=(), minval=0, maxval=len(self.injection_points),dtype=tf.int32)
        #Get the extracted layer name
        self.layer_name = self.injection_points[selected_injection]
        
    #Save the layer name to disable it later
    self.previous_injection = self.layer_name

    layer_activation(self)


class ClassesLayerPolicy(keras.callbacks.Callback):

    def __init__(self, CLASSES,extraction_frequency = 1.0, use_batch = False, mixed_callback = None,uniform_extraction=True):
        super().__init__()
        self.injection_points   = CLASSES.get_injection_points()
        self.CLASSES            = CLASSES
        self.uniform_extraction = uniform_extraction
        self.current_stack      = copy.deepcopy(self.injection_points)
        self.previous_injection = self.injection_points[0]
        self.extraction_frequency = extraction_frequency
        self.use_batch            = use_batch
        self.mixed_callback       = mixed_callback

    def on_train_batch_begin(self, epoch, logs=None):
        if self.use_batch:
            if self.mixed_callback != None and self.mixed_callback.golden != True:
                return
            selection_policy(self)

    def on_epoch_begin(self, epoch, logs=None):
        if not self.use_batch:
            selection_policy(self)
        if self.mixed_callback != None and self.mixed_callback.golden != True:
            self.CLASSES.disable_all()
        