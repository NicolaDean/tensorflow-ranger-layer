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

np.random.seed(42)

data_dir = '../datasets/GTSRB'
train_path = '../datasets/GTSRB/Train'
test_path = '../datasets/GTSRB/'

# Resizing the images to 32x32x3
IMG_HEIGHT = 32
IMG_WIDTH = 32
channels = 3

NUM_CATEGORIES = len(os.listdir(train_path))
NUM_CATEGORIES


data = pd.read_csv("../datasets/GTSRB/Test.csv")
print(data)

image_data = []
image_labels = []
for index, row in tqdm(data.iterrows(),total=data.shape[0]):
    path = row['Path']
    path = "../datasets/GTSRB/" + "/" + path
    image = cv2.imread(path)
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    image_data.append(np.array(resize_image))
    image_labels.append(row['ClassId'])

x_test = np.array(image_data)
y_test = np.array(image_labels)
print(x_test.shape, y_test.shape)

y_test = keras.utils.to_categorical(y_test, NUM_CATEGORIES)
print(y_test.shape)

x_test = x_test/255

model = tf.keras.models.load_model("../saved_models/vgg19_gtsrb")
model.summary()

model.evaluate(x_test,y_test)


#TRAINING
exit()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42, shuffle=True)

x_train = x_train/255 
x_val = x_val/255

print("X_train.shape", x_train.shape)
print("X_valid.shape", x_val.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_val.shape)

y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)

print(y_train.shape)
print(y_val.shape)

from tensorflow.keras.applications.vgg19 import VGG19
model2 = Sequential()
model2.add(VGG19(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT,IMG_WIDTH,channels)))
model2.add(BatchNormalization())
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dense(43, activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
history = model2.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_data=(x_val, y_val),workers=24)

model2.save("vgg19_gtsrb")