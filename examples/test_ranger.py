from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from enum import Enum
import sys


RANGER_MODULE_PATH = "../"

# appending a path
sys.path.append(RANGER_MODULE_PATH) #CHANGE THIS LINE

from custom_layers.ranger import *

ranger_layer = Ranger(name="ranger_test")
ranger_layer.set_ranger_mode(RangerModes.RangeTuning)
ranger_layer.set_ranger_policy(RangerPolicies.Clipper)
ranger_layer.set_ranger_granularity(RangerGranularity.Layer)


def fake_range_tuning(layer):
    print("="*40)
    print("===========Range Tuning================")
    print("="*40)

    layer.set_ranger_mode(RangerModes.RangeTuning)

    input = tf.constant([[1, 2, 3], [4, 5, 6],[7,8,9]],dtype=tf.float32)
    input = tf.expand_dims(input,0)
    out = layer(input) #Should be (1,9)

    input = tf.constant([[1, -10, 3], [4, 5, 6],[7,22,9]],dtype=tf.float32)
    input = tf.expand_dims(input,0)
    out = layer(input) #Should be (-10,22)

    input = tf.constant([[1, -12, 3], [4, 5, 6],[7,21,24]],dtype=tf.float32)
    input = tf.expand_dims(input,0)
    out = layer(input) #Should be (-10,22)
    layer.set_ranger_mode(RangerModes.Inference)


fake_range_tuning(ranger_layer)

#-----------------------------------------------
#--------------CLIPPING LAYER MODE--------------
#-----------------------------------------------
print("="*40)
ranger_layer.set_ranger_mode(RangerModes.Inference)
ranger_layer.set_ranger_policy(RangerPolicies.Clipper)
ranger_layer.set_ranger_granularity(RangerGranularity.Layer)

IMAGES = [[[1, -13, -23], [4, 25, 6],[7,24,2]],[[1, -1, -23], [4, 25, 6],[7,24,2]]]
input = tf.constant(IMAGES,dtype=tf.float32)
input = tf.expand_dims(input,0)
out = ranger_layer(input)
print("-------Before Clipping-----------")
tf.print(input)
print("-------After Clipping-----------")
tf.print(out)


#---------------------------------------------
#--------------RANGER LAYER MODE--------------
#---------------------------------------------
print("="*40)
ranger_layer.set_ranger_policy(RangerPolicies.Ranger)
ranger_layer.set_ranger_granularity(RangerGranularity.Layer)

input = tf.constant(IMAGES,dtype=tf.float32)
input = tf.expand_dims(input,0)
out = ranger_layer(input)
print("-------Before Ranger-----------")
tf.print(input)
print("-------After Ranger-----------")
tf.print(out)



#-----------------------------------------------
#--------------CLIPPING VALUE MODE--------------
#-----------------------------------------------
print("="*40)
ranger_layer.set_ranger_policy(RangerPolicies.Clipper)
ranger_layer.set_ranger_granularity(RangerGranularity.Value)
ranger_layer.reset_w()
fake_range_tuning(ranger_layer)
ranger_layer.print_w()

input = tf.constant(IMAGES,dtype=tf.float32)
input = tf.expand_dims(input,0)
out = ranger_layer(input)
print("-------Before Clipping by value-----------")
tf.print(input)
print("-------After Clipping by value-----------")
tf.print(out)

#-----------------------------------------------
#--------------RANGER VALUE MODE--------------
#-----------------------------------------------