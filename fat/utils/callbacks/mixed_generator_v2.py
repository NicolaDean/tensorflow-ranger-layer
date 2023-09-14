import tensorflow as tf
import keras
import copy
import sys
import random
import numpy as np
sys.path.append("./../../../model_helper/")

from model_helper.classes_model import *


class MixedGeneratorV2Obj(keras.callbacks.Callback):

    def __init__(self, num_epochs_switch):
        super().__init__()
        self.num_epochs_switch = num_epochs_switch
        self.golden = True
        self.curr_epochs = 0


    def on_epoch_end(self, epoch, logs=None):
        self.curr_epochs += 1
        if self.curr_epochs == self.num_epochs_switch:
            self.curr_epochs = 0
            self.golden = not self.golden