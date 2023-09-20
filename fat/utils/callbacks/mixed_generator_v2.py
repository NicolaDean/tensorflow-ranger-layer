import tensorflow as tf
import keras
import copy
import sys
import random
import numpy as np
sys.path.append("./../../../model_helper/")

from model_helper.classes_model import *


class MixedGeneratorV2Obj(keras.callbacks.Callback):

    def __init__(self, num_epochs_switch, v3 = False, f1_target = 0.0):
        super().__init__()
        self.num_epochs_switch = num_epochs_switch
        self.golden = True
        self.v3 = v3
        self.regen_golden = False
        self.f1_target = f1_target
        self.f1_current = 1.0
        

    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.num_epochs_switch == 0:
            self.golden = not self.golden
        if epoch%(self.num_epochs_switch*2) == 0 and self.v3:
            self.regen_golden = True
        
