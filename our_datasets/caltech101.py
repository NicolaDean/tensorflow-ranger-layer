
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pathlib
import sys
np.random.seed(42)

directory = str(pathlib.Path(__file__).parent.absolute())

data_dir   = directory + '/Caltech101'
train_path = directory + '/Caltech101/Train'
test_path  = directory + '/Caltech101/Test'

# Resizing the images to 32x32x3
IMG_HEIGHT = 32
IMG_WIDTH = 32
channels = 3

NUM_CATEGORIES = len(os.listdir(train_path))
NUM_CATEGORIES

def load_train(shape = (IMG_HEIGHT, IMG_WIDTH)):
    data = pd.read_csv("../our_datasets/Caltech101/Train.csv")
    print(data)

    image_data = []
    image_labels = []
    for index, row in tqdm(data.iterrows(),total=data.shape[0]):
        path = row['Path']
        path = "../our_datasets/Caltech101/" + "/" + path
        image = cv2.imread(path)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize(shape)
        image_data.append(np.array(resize_image))
        image_labels.append(row['ClassId'])

    x_train, x_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

    x_train = np.array(x_train, dtype = float)
    x_val = np.array(x_val, dtype = float)
    y_train = np.array(y_train, dtype = float)
    y_val = np.array(y_val, dtype = float)    
    print(y_train.shape)
    print(y_val.shape)


    return x_train, x_val[:500], y_train, y_val[:500]

def load_test(shape = (IMG_HEIGHT, IMG_WIDTH)):
    data = pd.read_csv("../our_datasets/Caltech101/Test.csv")
    print(data)

    image_data = []
    image_labels = []
    for index, row in tqdm(data.iterrows(),total=data.shape[0]):
        path = row['Path']
        path = "../our_datasets/Caltech101/" + "/" + path
        image = cv2.imread(path)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize(shape)
        image_data.append(np.array(resize_image))
        image_labels.append(row['ClassId'])

    x_test = np.array(image_data)
    y_test = np.array(image_labels)
    print(x_test.shape, y_test.shape)

    y_test = keras.utils.to_categorical(y_test, NUM_CATEGORIES)
    print(y_test.shape)

    x_test = x_test/255

    return x_test,y_test