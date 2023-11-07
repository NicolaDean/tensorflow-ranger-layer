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
from datasets import self_drive

#x_train,y_train,x_val,y_val = self_drive.loadDataWithClassDict(256)

'''
train_gen, valid_gen, train_size, val_size = self_drive.GET_DATA_GENERATOR(256,batch_size=32)

x,y = next(train_gen)

print(f'Shapes: {x.shape},{y.shape}')
## instanctiating model
inputs = tf.keras.layers.Input((256, 256, 3))
unet = GiveMeUnet(inputs, droupouts= 0.07)
unet.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )

unet.summary()
            
## trainign our model
retVal = unet.fit(train_gen,
                steps_per_epoch=max(1, train_size//32),
                validation_data=valid_gen,
                validation_steps=max(1, val_size//32),
                epochs = 150)

unet.save("../saved_models/unet_self_drive_v2")
'''

inputs = tf.keras.layers.Input((256, 256, 3))
unet = GiveMeUnet(inputs, droupouts= 0.07)
unet.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )

unet.summary()

train_generator_fn,val_generator_fn = self_drive.get_generator()

epochs=50
steps=int(369//32)
valid_steps=int(100//32)

hisotry=unet.fit(train_generator_fn(),
        validation_data=val_generator_fn(),
        steps_per_epoch=steps,
        validation_steps=valid_steps,
        epochs=epochs,)

unet.save("../saved_models/unet_self_drive_v2")