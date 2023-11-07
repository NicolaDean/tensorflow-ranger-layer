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


# directory reach
LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *


from models.unet_model import *
from utils.fat_generic import run_fat_experiment

# directory reach
LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *
from our_datasets import pet
from our_datasets import self_drive


injection_point = []
injection_point += [f'conv2d_{i}' for i in range(1,19)]

def unet_fat():
        
        model = tf.keras.models.load_model("../saved_models/unet_self_drive_v2")

        model.summary()

        train_generator_fn,place_holder = self_drive.get_generator(batch_size=32,not_valid=True)
        place_holder,val_generator_fn   = self_drive.get_generator(batch_size=1,not_train=True)

        train_generator_fn = train_generator_fn()
        val_generator_fn   = val_generator_fn()

        train_size = 369 // 32
        valid_size = 100 // 1
        
        run_fat_experiment(model=model,
                           train_gen=train_generator_fn,
                           valid_gen=val_generator_fn,
                           train_size=train_size,
                           valid_size=valid_size,
                           frequency=0.5,
                           epochs=20,
                           injection_point = injection_point
                           )

unet_fat()