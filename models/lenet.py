from tensorflow import keras
from keras import Sequential
import tensorflow as tf

layers = keras.layers

def LeNet(input_shape):
    model = Sequential()
    model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=input_shape))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(16, 5,strides=(1,1), activation='tanh'))
    model.add(layers.AveragePooling2D(2,strides=(1,1)))
    model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(120, 5,strides=(1,1), activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(10, activation='softmax'))
    return model