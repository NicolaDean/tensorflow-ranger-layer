import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D,Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random
import warnings
from PIL import Image

import sys
import pathlib
directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + "/../")

warnings.filterwarnings("ignore")
from our_datasets import stearing_angle

#(200,66)
SHAPE = (64,64)
IN_SHAPE = (SHAPE[1],SHAPE[0],3)

def nvidiaModel(shape = IN_SHAPE):
  input = Input(shape = shape)

  model = Conv2D(24,(5,5),strides=(2,2),activation="elu")(input)
  model = Conv2D(36,(5,5),strides=(2,2),activation="elu")(model)
  model = Conv2D(48,(5,5),strides=(2,2),activation="elu")(model)
  model = Conv2D(64,(3,3),activation="elu")(model)
  model = Conv2D(64,(3,3),activation="elu")(model)
  model = Dropout(0.5)(model)
  
  model = GlobalAveragePooling2D()(model)
  
  model = Dense(128,activation="elu")(model)
  model = Dropout(0.5)(model)
  
  model = Dense(64,activation="elu")(model)
  model = Dropout(0.5)(model)
  
  model = Dense(16,activation="elu")(model)
  model = Dropout(0.5)(model)
  
  out = Dense(1)(model)

  model = keras.Model(inputs=input, outputs=out)

  model.compile(loss='mean_absolute_error',optimizer=Adam(lr=1e-3),metrics=['accuracy','mean_absolute_error'])
  
  return model

model =  nvidiaModel()
model.summary()

x_train,x_val,y_train,y_val = stearing_angle.load_data(SHAPE) 

x_train = x_train/255
x_val   = x_val/255

model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val))

model.save('Dave_STEERING_ANGLE_v4')

exit()
history = model.fit_generator(batch_generator(x_train, y_train, 100, 1),
                                  steps_per_epoch=300, 
                                  epochs=10,
                                  validation_data=batch_generator(x_val, y_val, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)

