from tensorflow import keras
from keras import Sequential
import tensorflow as tf

layers = keras.layers

def LeNet(input_shape):
    model = Sequential()
    model.add(layers.Conv2D(6, 3, activation='relu', input_shape=input_shape))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Conv2D(16, 3,strides=(1,1), activation='relu'))
    model.add(layers.AveragePooling2D(2,strides=(1,1)))
    model.add(layers.Conv2D(120, 3,strides=(1,1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model