import tensorflow as tf
import keras
import copy
import sys
import random
import numpy as np
sys.path.append("./../../../model_helper/")

from model_helper.classes_model import *

class ClassesLayerPolicy(keras.callbacks.Callback):

    def __init__(self, CLASSES,uniform_extraction=True):
        super().__init__()
        self.injection_points   = CLASSES.get_injection_points()
        self.CLASSES            = CLASSES
        self.uniform_extraction = uniform_extraction
        self.current_stack      = copy.deepcopy(self.injection_points)
        self.previous_injection = self.injection_points[0]

    def on_train_batch_begin(self, epoch, logs=None):

        #Disable previously selected injection point:
        layer = CLASSES_HELPER.get_layer(self.model,self.previous_injection,verbose=False)
        layer.set_mode(ErrorSimulatorMode.disabled)  #Enable the Selected Injection point

        #IF WE CHOSED TO UNIFORMLY ASSIGN INJECTION POINTS USE THE EXTRACTION STACK
        if self.uniform_extraction:
            #self.previous_injection = random.shuffle(self.current_stack)
            random.shuffle(self.current_stack)
            selected_injection = self.current_stack.pop()
            self.previous_injection = selected_injection

            if not self.current_stack:
                self.current_stack = copy.deepcopy(self.injection_points)
                print("Refill Stack of injection points")
        #IF WE CHOSED TO PICK AT RANDOM SIMPLY EXTRACT THE INDEX
        else:
            #Select a random Injection point from the list
            selected_injection = tf.random.uniform(shape=(), minval=0, maxval=len(self.injection_points),dtype=tf.int32)
      
            selected_injection = self.injection_points[selected_injection]
            self.previous_injection = selected_injection

        #Enable the selected injection point:
        layer = CLASSES_HELPER.get_layer(self.model,selected_injection,verbose=False)
        layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point

        #Print the Name of the activated Layer
        #print("Injection point [ {} ] Activated for epoch: {};".format(selected_injection, epoch))