from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import copy
import sys

from enum import IntEnum

class RangerModes(IntEnum):
    Training        = 0,    #Return input as is
    RangeTuning     = 1,    #Comput Layer Domain and return input as is
    Inference       = 2,    #Apply Clipping or threshold to input
    Disabled        = 3,    #Return input as is


#these policy names are taken from the content of "Towards a Safety Case for Hardware Fault Tolerance in Convolutional Neural 
#Networks Using Activation Range Supervision" paper
class RangerPolicies(IntEnum):
    Clipper         = 0,        #clip to 0 outliers
    Ranger          = 1,        #saturate to range bounds outliers

#different granularities refer to different 
class RangerGranularity(IntEnum):
    Layer           = 0,        #Each Layer has a single min/max value
    Value           = 1,        #Each single value in each layer has a min/max (eg each pixel)

class Ranger(keras.layers.Layer):
    def __init__(self, name):
        super(Ranger, self).__init__(name = name)
        self.mode        = tf.Variable(int(RangerModes.Training),trainable=False)
        self.policy      = tf.Variable(int(RangerPolicies.Clipper),trainable=False)#RangerPolicies.Clipper
        self.granularity = tf.Variable(int(RangerGranularity.Layer),trainable=False)#RangerGranularity.Layer
    
    '''
    Set the Mode of the ranger layer among the one described in RangerModes class
    '''
    def set_ranger_mode(self,mode:RangerModes):
        self.mode.assign(int(mode))
    
    def set_ranger_policy(self, policy:RangerPolicies):
        self.policy.assign(int(policy))
        #self.policy = policy
    
    def set_ranger_granularity(self, granularity:RangerGranularity):
        self.granularity.assign(int(granularity))
        #self.granularity = granularity

    def is_mode(self,mode:RangerModes):
        return self.mode == tf.constant([[int(mode)]])
    def is_granularity(self,mode:RangerGranularity):
        return self.granularity == tf.constant([[int(mode)]])
    
    def print_layer_config(self):
        tf.print("Granularity: ",RangerGranularity(self.granularity).name)
        tf.print("Policy     : ",RangerPolicies(self.policy).name)
        tf.print("Mode       : ",RangerModes(self.mode).name)
    
    def reset_weights(self,input_shape):
        
        if(input_shape[0] == None):
            in_shape = (1,input_shape[1],input_shape[2],input_shape[3])
        else:
            in_shape = input_shape
        
        #self.input_shape = input_shape
        #self.print_layer_config()

        def Layer_granularity():
            range_max = tf.constant(tf.dtypes.float32.min)
            range_min = tf.constant(tf.dtypes.float32.max)
            return (range_min,range_max)
           
        def Value_granularity():
            range_max = tf.fill(in_shape,tf.dtypes.float32.min)
            range_min = tf.fill(in_shape,tf.dtypes.float32.max)
            return (range_min,range_max)
        
        is_layer_mode   = self.is_granularity(RangerGranularity.Layer)
        ranges_w        = tf.cond(is_layer_mode,Layer_granularity,Value_granularity)
        self.w          = tf.Variable(initial_value = ranges_w, trainable = False)
        #tf.print(f"RANGE W has shape: {self.w.shape}")

    def build(self, input_shape):
        self.shape = input_shape
        self.reset_weights(input_shape)

    '''
    Compute the Layer Domain of values
    '''
    def range_tuning_v2(self,inputs):
        range_min = self.w[0]
        range_max = self.w[1]

        def Layer_granularity():
            inputs_min = tf.math.reduce_min(inputs)
            inputs_max = tf.math.reduce_max(inputs)
            return inputs_min,inputs_max
        def Value_granularity():
            return inputs,inputs
        
        is_layer_mode = self.is_granularity(RangerGranularity.Layer)
        inputs_min,inputs_max = tf.cond(is_layer_mode,Layer_granularity,Value_granularity)

        range_min = tf.where(tf.less_equal   (inputs_min,range_min),inputs_min,range_min)
        range_max = tf.where(tf.greater_equal(inputs_max,range_max),inputs_max,range_max)

        #tf.print("MINMAX: ",inputs_min,inputs_max)
        #tf.print("RANGE :\n", range_min,range_max, output_stream=sys.stdout)

        tmp_w = []
        tmp_w.append(range_min)
        tmp_w.append(range_max)

        w = tf.stack(tmp_w) #Crea un array del tipo [range_min,range_max] in tensorflow
        self.w.assign(w)
        
        return inputs
    
    '''
    Apply the threshold clipping or threshold
    '''
    def apply_range_threshold_v2(self,inputs):
        #TODO CONVERT CODE TO USE https://www.tensorflow.org/api_docs/python/tf/clip_by_value in the Clipper version
        range_min   = self.w[0]
        range_max   = self.w[1]

        upper_threshold = tf.greater_equal(range_max,inputs)
        lower_threshold = tf.less_equal(range_min,inputs)
        

        def Clipper_Policy():
            #tf.print("Clipper")

            #outputs = tfp.math.clip_by_value_preserve_gradient(inputs, clip_value_min=range_min, clip_value_max=range_max)

            in_range    = tf.logical_and(lower_threshold, upper_threshold)
            outputs     = tf.where(in_range,inputs,0)
            return outputs
        
        def Ranger_Policy():
            #tf.print("Ranger")
            outputs = tf.where(lower_threshold,inputs ,range_min)
            outputs = tf.where(upper_threshold,outputs,range_max)
            return outputs
        
        def Default_Policy():
            return inputs
        
        cases = {int(RangerPolicies.Clipper):Clipper_Policy,int(RangerPolicies.Ranger):Ranger_Policy}
        index = tf.convert_to_tensor(self.policy)
        
        return tf.switch_case(index,cases,default = Default_Policy)
    
    def get_grad_threshold(self, inputs):
        #in this case policy does not impact on the gradient: when out of range gradient = 0 

        range_min   = self.w[0]
        range_max   = self.w[1]

        upper_threshold = tf.greater_equal(range_max, inputs)
        lower_threshold = tf.less_equal(range_min,inputs)
        in_range    = tf.logical_and(lower_threshold, upper_threshold)
        return tf.cast(in_range, tf.float32)


    def print_w(self):
        tf.print(self.w)

    def reset_w(self):
        print(f"RESET {self.name} Ranges")
        self.reset_weights(self.shape)

    @tf.custom_gradient
    def call(self, inputs):
        
        switch_cases = {
            int(RangerModes.Training)   :lambda: inputs,
            int(RangerModes.RangeTuning):lambda: self.range_tuning_v2(inputs),
            int(RangerModes.Inference)  :lambda: self.apply_range_threshold_v2(inputs),
            int(RangerModes.Disabled)   :lambda: inputs
        }

        index = tf.convert_to_tensor(self.mode)
        def grad(upstream):
            switch_cases_grad = {
            int(RangerModes.Training)   :lambda: tf.ones_like(inputs),
            int(RangerModes.RangeTuning):lambda: tf.ones_like(inputs),
            int(RangerModes.Inference)  :lambda: self.get_grad_threshold(inputs),
            int(RangerModes.Disabled)   :lambda: tf.ones_like(inputs)
            }
            return tf.multiply(tf.switch_case(index,switch_cases_grad), upstream)
        return tf.switch_case(index,switch_cases), grad
        #return tf.switch_case(index,switch_cases)

                                         


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
            
            
            tf.print("RANGE :", range_min,range_max, output_stream=sys.stdout)

            tmp_w.append(range_min)
            tmp_w.append(range_max)

            w = tf.stack(tmp_w) #Crea un array del tipo [range_min,range_max] in tensorflow
            self.w.assign(w)

            #let intermediate pass through without any modifications
            return inputs
    
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