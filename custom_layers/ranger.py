from tensorflow import keras
import tensorflow as tf
import numpy as np
import copy
import sys

from enum import IntEnum

class RangerModes(IntEnum):
    Training        = 1,
    RangeTuning     = 2,
    Inference       = 3,
    Disabled        = 4,


#these policy names are taken from the content of "Towards a Safety Case for Hardware Fault Tolerance in Convolutional Neural 
#Networks Using Activation Range Supervision" paper
class RangerPolicies(IntEnum):
    Clipper         = 1,        #clip to 0 outliers
    Ranger          = 2,        #saturate to range bounds outliers

#different granularities refer to different 
class RangerGranularity(IntEnum):
    Layer           = 1,
    Value           = 2,

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

class Ranger(keras.layers.Layer):
    def __init__(self, name):
        super(Ranger, self).__init__(name = name)
        self.mode = tf.Variable(int(RangerModes.Training),trainable=False)
        self.policy = RangerPolicies.Clipper
        self.granularity = RangerGranularity.Layer
    
    def set_ranger_mode(self,mode:RangerModes):
        self.mode.assign(int(mode))
    
    def set_ranger_policy(self, policy:RangerPolicies):
        self.policy = policy
    
    def set_ranger_granularity(self, granularity:RangerGranularity):
        self.granularity = granularity

    def build(self, input_shape):
        #range_max = tf.experimental.numpy.full(input_shape[1:], tf.dtypes.float32.min, dtype = tf.float32)
        #range_min = tf.experimental.numpy.full(input_shape[1:], tf.dtypes.float32.max, dtype = tf.float32)

        #Per non usare features sperimentali
        
        if self.granularity == RangerGranularity.Layer:
            range_max = tf.constant(tf.dtypes.float32.min)
            range_min = tf.constant(tf.dtypes.float32.max)
        else:
            range_max = tf.fill(input_shape[1:],tf.dtypes.float32.min)
            range_min = tf.fill(input_shape[1:],tf.dtypes.float32.max)

        self.w = tf.Variable(initial_value = (range_min, range_max), trainable = False)
        self.record = False         #state if the range can be evaluated -> TRUE in the middle between training and final inference
    
    '''
    Compute the Layer Domain of values
    '''
    def range_tuning(self,inputs):
            #tf.print("RangeTuning",output_stream=sys.stdout)
            inp = tf.reshape(inputs, inputs.shape[1:])
            #update ranges -> no effect on the final result
            #w = self.get_weights()[0]

            range_min = self.w[0]
            range_max = self.w[1]
            #tf.print("Range :", range_min,range_max, output_stream=sys.stdout)
            bool_max        = tf.greater_equal(range_max,inp)
            bool_max        = tf.cast(bool_max, tf.float32)
            not_bool_max    = tf.ones(bool_max.shape) - bool_max
            range_max       = bool_max * range_max + not_bool_max * inp

            bool_min        = tf.less_equal(range_min,inp)
            bool_min        = tf.cast(bool_min, tf.float32)
            not_bool_min    = tf.ones(bool_min.shape) - bool_min
            range_min       = bool_min*range_min + not_bool_min*inp

            tmp_w = []
            
            if self.granularity == RangerGranularity.Layer:
                #global_max = tf.max(range_max)
                #range_max  = tf.fill(range_max.shape, global_max)
                global_max = tf.math.reduce_max(range_max)
                range_max = global_max

                #global_min = tf.min(range_min)                
                #range_min  = tf.fill(range_min.shape, global_min)
                global_min = tf.math.reduce_min(range_min)
                range_min = global_min
            
            
            #tf.print("RANGE :", range_min,range_max, output_stream=sys.stdout)

            tmp_w.append(range_min)
            tmp_w.append(range_max)

            w = tf.stack(tmp_w) #Crea un array del tipo [range_min,range_max] in tensorflow
            self.w.assign(w)

            #let intermediate pass through without any modifications
            return inputs
    
    '''
    Apply the threshold clipping or threshold
    '''
    def apply_range_threshold(self,inputs):
            #tf.print("Inference",output_stream=sys.stdout)
            #final inference
            range_min   = self.w[0]
            range_max   = self.w[1]
            #bool_max    = range_max >= inputs
            #bool_min    = range_min <= inputs

            bool_max    = tf.greater_equal(range_max,inputs)    #Work also if shape or range is [1,1]
            bool_min    = tf.less_equal(range_min,inputs)

            bool_max    = tf.cast(bool_max, tf.float32)
            bool_min    = tf.cast(bool_min, tf.float32)

            merged_mask = bool_max * bool_min #LOGIC AND => 1 * 1 = 1 /  1 * 0 = 0 / 0 * 1 = 0 / 0 * 0 = 0
            #merged_mask = tf.matmul(bool_max,bool_min)

            if self.policy == RangerPolicies.Clipper:
                #return tf.matmul(merged_mask,inputs)
                return merged_mask * inputs
            elif self.policy == RangerPolicies.Ranger:
                #return tf.matmul(merged_mask,inputs) + tf.matmul(bool_max,range_max) + tf.matmul(bool_min , range_min)
                return merged_mask * inputs + bool_max * range_max + bool_min * range_min
    
    '''
    Create a TF graph to switch between inference and range threshold
    '''
    def ranger_mode(self,inputs):
        range       = lambda: self.range_tuning(inputs)
        inference   = lambda: self.apply_range_threshold(inputs)

        return tf.cond(self.mode == tf.constant([[int(RangerModes.RangeTuning)]]),
                       true_fn  = range,
                       false_fn = inference)
    
    def call(self, inputs):
        true_fn     = lambda: inputs
        false_fn    = lambda: self.ranger_mode(inputs)
        
        return tf.cond(self.mode == tf.constant([[int(RangerModes.Disabled)]]) or self.mode == tf.constant([[int(RangerModes.Training)]]),
                true_fn  = true_fn,
                false_fn = false_fn
        )
                                         

        