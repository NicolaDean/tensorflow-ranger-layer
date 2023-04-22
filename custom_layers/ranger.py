import tensorflow as tf
import numpy as np
import copy
import sys

from enum import Enum

class RangerModes(Enum):
    Training        = 1,
    RangeTuning     = 2,
    Inference       = 3,
    Disabled        = 4,

'''

class Ranger(tf.keras.layers.Layer):
    def __init__(self, name):
        super(Ranger, self).__init__(name = name)
    
    def build(self, input_shape):
        self.max = tf.experimental.numpy.full(input_shape[1:], tf.dtypes.float32.min, dtype = tf.float32)
        self.min = tf.experimental.numpy.full(input_shape[1:], tf.dtypes.float32.max, dtype = tf.float32)
        self.record = False         #state if the range can be evaluated -> TRUE in the middle between training and final inference
    
    def call(self, inputs):
        if self.record:
            inp = tf.reshape(inputs, inputs.shape[1:])
            #update ranges -> no effect on the final result
            bool_max = self.max >= inp
            
            bool_max = tf.cast(bool_max, tf.float32)
            not_bool_max = tf.ones(bool_max.shape) - bool_max
            self.max = bool_max*self.max + not_bool_max*inp

            bool_min = self.min <= inp
            bool_min = tf.cast(bool_min, tf.float32)
            not_bool_min = tf.ones(bool_min.shape) - bool_min
            self.min = bool_min*self.min + not_bool_min*inp

            #let intermediate pass through without any modifications
            return inputs
        else:

            if self.trainable:
                #during training
                return inputs
            else:
                #final inference
                bool_max = self.max >= inputs
                bool_min = self.min <= inputs
                bool_max = tf.cast(bool_max, tf.float32)
                bool_min = tf.cast(bool_min, tf.float32)
                merged_mask = bool_max * bool_min
                return merged_mask * inputs
    
    def set_weights(self, values):
        self.min = values[0]
        self.max = values[1]
    
    def get_weights(self):
        return (self.min, self.max)
'''

class Ranger(tf.keras.layers.Layer):
    def __init__(self, name):
        super(Ranger, self).__init__(name = name)
        self.mode = RangerModes.Training
    
    def set_ranger_mode(self,mode:RangerModes):
        self.mode = mode

    def build(self, input_shape):

        range_max = tf.experimental.numpy.full(input_shape[1:], tf.dtypes.float32.min, dtype = tf.float32)
        range_min = tf.experimental.numpy.full(input_shape[1:], tf.dtypes.float32.max, dtype = tf.float32)
        self.w = tf.Variable(initial_value = (range_min, range_max), trainable = False)
        self.record = False         #state if the range can be evaluated -> TRUE in the middle between training and final inference
    
    def call(self, inputs):
        if self.mode == RangerModes.Disabled or self.mode == RangerModes.Training:
            return inputs
        
        elif self.mode == RangerModes.RangeTuning:
            inp = tf.reshape(inputs, inputs.shape[1:])
            #update ranges -> no effect on the final result
            w = self.get_weights()[0]

            range_min = w[0]
            range_max = w[1]

            bool_max = range_max >= inp
            bool_max = tf.cast(bool_max, tf.float32)
            not_bool_max = tf.ones(bool_max.shape) - bool_max
            range_max = bool_max*range_max + not_bool_max*inp

            bool_min = range_min <= inp
            bool_min = tf.cast(bool_min, tf.float32)
            not_bool_min = tf.ones(bool_min.shape) - bool_min
            range_min = bool_min*range_min + not_bool_min*inp

            w = np.array([range_min, range_max])

            self.set_weights([w])

            #let intermediate pass through without any modifications
            return inputs
        elif self.mode == RangerModes.Inference:
            #final inference
            range_min = self.w[0]
            range_max = self.w[1]
            bool_max = range_max >= inputs
            bool_min = range_min <= inputs
            bool_max = tf.cast(bool_max, tf.float32)
            bool_min = tf.cast(bool_min, tf.float32)
            merged_mask = bool_max * bool_min
            return merged_mask * inputs
        else:
            return inputs

        