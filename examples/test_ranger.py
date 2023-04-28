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


input = tf.constant([[1, 2, 3], [4, 5, 6],[7,8,9]],dtype=tf.float32)
input = tf.expand_dims(input,0)
out = ranger_layer(input) #Should be (1,9)


input = tf.constant([[1, -10, 3], [4, 5, 6],[7,22,9]],dtype=tf.float32)
input = tf.expand_dims(input,0)
out = ranger_layer(input) #Should be (-10,22)

input = tf.constant([[1, -12, 3], [4, 5, 6],[7,21,9]],dtype=tf.float32)
input = tf.expand_dims(input,0)
out = ranger_layer(input) #Should be (-10,22)

ranger_layer.set_ranger_mode(RangerModes.Inference)

input = tf.constant([[1, -13, -23], [4, 25, 6],[7,24,9]],dtype=tf.float32)
input = tf.expand_dims(input,0)
out = ranger_layer(input)
print("-------Before Clipping-----------")
tf.print(input)
print("-------After Clipping-----------")
tf.print(out)