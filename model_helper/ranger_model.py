import tensorflow as tf
from enum import Enum
import sys

RANGER_MODULE_PATH = "../"

# appending a path
sys.path.append(RANGER_MODULE_PATH) #CHANGE THIS LINE

from custom_layers.ranger import *

class RangerModel(tf.keras.Model):

    def __init__(self,input, output,name="RangerModel"):
        super().__init__(input,output,name)

    '''
    Find all Ranger layer in the network and set theyr mode to a specific one
    '''
    def set_ranger_mode(self,mode:RangerModes):
        layers = [layer for layer in self.layers]
        for l in layers:
            if isinstance(l,Ranger):
                l.set_ranger_mode(mode)

    '''
    Take in input a classic CNN and add ranger layer for each Convolution
    '''
    def convert_model(model: tf.keras.Model):
        #TODO NOT IMPLEMENTED YET
        layers = [layer for layer in model.layers]
        for l in layers:
            if isinstance(l,tf.layers.Conv2D):
                print("Convolution:")
        return 1
    

    def call(self, inputs):
        #TODO
        super().call(inputs)



'''
IDEAS ON HOW IT SHOULD WORK

input = InputLayer(...)

x = Conv2D(...)(input)
x = Conv2D(...)(x)
....
x = Conv2D(...)(x)
output = Dense(...)(x)

model = RangerModel(input, output, "My_ranger_Model")
model.compile(...)


model.set_ranger_mode(RangerModes.Training)

[... TRAIN THE MODEL ...]

model.set_ranger_mode(RangerModes.RangeTuning)

[... Run one epochs  to extract domain...]

model.set_ranger_mode(RangerModes.Inference)

[... Now we can integrate Classes layers and test Ranger stuff ...]

'''