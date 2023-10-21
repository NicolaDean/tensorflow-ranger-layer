import tensorflow as tf
import keras
import copy
import sys
import random
import numpy as np
sys.path.append("./../../../model_helper/")

from model_helper.classes_model import *

from .random_injection import layer_activation

RANDOM_EXTRACTION       = 0
UNIFORM_EXTRACTION      = 1
ORDERED_EXTRACTION_CRE  = 2
ORDERED_EXTRACTION_DEC  = 3
ORDERED_FREEZE_CRE      = 4
ORDERED_FREEZE_DEC      = 5

#Random = 0  => Ordine Casuale
#Rangom = -1 => Ordine Decrescente
#Random = +1 => Ordine Crescente
def selection_policy(self,epoch):
    #Disable previously selected injection point:
    layer = CLASSES_HELPER.get_layer(self.model,self.previous_injection,verbose=False)
    layer.set_mode(ErrorSimulatorMode.disabled)  #Enable the Selected Injection point
    
    
    #IF WE CHOSED TO PICK AT RANDOM SIMPLY EXTRACT THE INDEX
    if self.extraction_type == RANDOM_EXTRACTION: #self.type
        #Select a random Injection point from the list
        selected_injection = tf.random.uniform(shape=(), minval=0, maxval=len(self.injection_points),dtype=tf.int32)
        #Get the extracted layer name
        self.layer_name = self.injection_points[selected_injection]
    #IF WE CHOSED TO UNIFORMLY ASSIGN INJECTION POINTS USE THE EXTRACTION STACK
    elif self.extraction_type == UNIFORM_EXTRACTION:
        #Shuffle the stack
        random.shuffle(self.current_stack)
        #Extract the selected layer
        self.layer_name         = self.current_stack.pop()
        
        if not self.current_stack:
            #Regenerate the stack when its empty
            self.current_stack = copy.deepcopy(self.injection_points)
    #IF WE CHOSED TO DO SOME EPOCHS FOR EACH LAYER IN ORDER OF THE MODEL CRESCENT
    elif self.extraction_type == ORDERED_EXTRACTION_CRE:
        #print(f"CREE: [{epoch}] => [{self.epoch_trigger}]" )
        if epoch % self.epoch_trigger == 0:
            #print("TRIGGERED")
            #Get the extracted layer name
            self.layer_name = self.injection_points[self.selected_injection]

            #TODO INITIALIZE THE VARIABLE
            self.selected_injection += 1

            if self.selected_injection >= len(self.injection_points):
                self.selected_injection = 0
    #IF WE CHOSED TO DO SOME EPOCHS FOR EACH LAYER IN ORDER OF THE MODEL DECRESCENT
    elif self.extraction_type == ORDERED_EXTRACTION_DEC:
        if epoch % self.epoch_trigger == 0:
            #Get the extracted layer name
            self.layer_name = self.injection_points[self.selected_injection]

            #TODO INITIALIZE THE VARIABLE
            self.selected_injection -= 1

            if self.selected_injection < 0:
                self.selected_injection = len(self.injection_points) - 1
    else:
        print(f"\033[0;31mERROR - PLEASE SELECT  A VALID LAYER EXTRACTION POLICY\033[0m")
        exit()
       
        
        
    #Save the layer name to disable it later
    self.previous_injection = self.layer_name

    if self.extraction_type <= UNIFORM_EXTRACTION:
        layer_activation(self)


class ClassesLayerPolicy(keras.callbacks.Callback):

    def __init__(self, CLASSES,extraction_frequency = 1.0, use_batch = False, mixed_callback = None,extraction_type=UNIFORM_EXTRACTION,epoch_trigger=1):
        super().__init__()
        self.injection_points   = CLASSES.get_injection_points()
        self.CLASSES            = CLASSES
        self.extraction_type    = extraction_type
        self.current_stack      = copy.deepcopy(self.injection_points)
        self.previous_injection = self.injection_points[0]
        self.extraction_frequency = extraction_frequency
        self.use_batch            = use_batch
        self.mixed_callback       = mixed_callback

        if self.extraction_type == ORDERED_EXTRACTION_CRE or self.extraction_type == ORDERED_EXTRACTION_DEC:
            self.use_batch          = False #FORCE TO USE EPOCHS WITH THIS STRATEGY
            self.selected_injection = 0
            self.epoch_trigger      = epoch_trigger


    def on_train_batch_begin(self, epoch, logs=None):
        if self.use_batch:
            if self.mixed_callback != None and self.mixed_callback.golden != True:
                return
            selection_policy(self,epoch)
        
        if self.extraction_type > UNIFORM_EXTRACTION:
            layer_activation(self)


    def on_epoch_begin(self, epoch, logs=None):
        if not self.use_batch:
            selection_policy(self,epoch)
        if self.mixed_callback != None and self.mixed_callback.golden != True:
            self.CLASSES.disable_all()
        