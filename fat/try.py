import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses


LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *

sys.path.append("./")

#LeNet + Mnist
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

#Original LeNet model receives 32 by 32 images, thus 28 by 28 MNIST images are padded with zeros and 8-bit (0–255 range) pixel values are scaled between 0 and 1
x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255

x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)

x_val = x_train[-2000:,:,:] 
y_val = y_train[-2000:] 
x_train = x_train[:-2000,:,:] 
y_train = y_train[:-2000]

model = models.Sequential()
model.add(layers.Conv2D(6, 3, activation='tanh', input_shape=x_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 3, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 3, activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.load_weights("lenet.h5")


ranger = RANGER_HELPER(model)
ranger.convert_model()
ranger.set_ranger_mode(mode = RangerModes.RangeTuning)
ranger.tune_model_range(x_train)
ranger.set_ranger_mode(mode = RangerModes.Inference)

model = ranger.get_model()
model.summary()

layer = model.layers[4]

classes = CLASSES_HELPER(model)

classes.convert_model(10)
classes.get_model().summary()
classes.disable_all()
classes.set_mode("conv2d_1", ErrorSimulatorMode.enabled)
'''


layer = model.layers[3]

image = np.expand_dims(x_train[0], axis = 0)
int_res = model.layers[0](image)
int_res = model.layers[1](int_res)
int_res = model.layers[2](int_res)

with tf.GradientTape() as tape:
    tape.watch(int_res)
    final_res = layer(int_res)

dy_dx = tape.gradient(final_res, int_res)  
print(dy_dx)
'''
model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))





