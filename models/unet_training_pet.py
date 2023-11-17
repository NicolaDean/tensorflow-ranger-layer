## Making essential imports
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm
import sys

from unet_model import *

# directory reach
LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *

from our_datasets import pet

IMG_SHAPE = (128,128)
IN_SHAPE = (128,128,3)
x_train,x_val,y_train,y_val = pet.load_data(IMG_SHAPE)

print(f'TRAINING => X: [{x_train.shape}], Y: [{y_train.shape}]')
print(f'VALIDATION => X: [{x_val.shape}], Y: [{y_val.shape}]')

inputs = tf.keras.layers.Input(IN_SHAPE)
unet = GiveMeUnet(inputs, droupouts= 0.07,output_channels=3)
unet.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )

unet.summary()


model_history = unet.fit(x_train,y_train,epochs=30,validation_data=(x_val,y_val))

unet.save("../saved_models/unet_pet_classic")