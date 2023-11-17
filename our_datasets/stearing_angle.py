import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D,Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random
import warnings
from PIL import Image
import pathlib
warnings.filterwarnings("ignore")

datadir = str(pathlib.Path(__file__).parent.absolute())

columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(f'{datadir}/car_simulator_dataset/driving_log.csv', names = columns)

data.head()

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)


num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)


center = (bins[:-1]+ bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), \
(samples_per_bin, samples_per_bin))

print('total data:', len(data))

remove_list = []
for j in range(num_bins):
  list_ = []
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)
print('removed:', len(remove_list))

data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))

hist, _ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), \
(samples_per_bin, samples_per_bin))

from tqdm import tqdm

print(data.iloc[1])
IN_SHAPE=(200, 66)

def load_img(path,shape=IN_SHAPE):
    img = cv2.imread(path)
    image_fromarray = Image.fromarray(img, 'RGB')
    resize_image = image_fromarray.resize(shape)
    
    return np.array(resize_image)

def load_img_steering(datadir, shape=IN_SHAPE):
  images = []
  steering = []
  
  for i in tqdm(range(len(data))):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    
    # center image append
    img_path = os.path.join(datadir, center.strip())
    img = load_img(img_path,shape)
    images.append(img)
    angle = float(indexed_data[3])
    angle = (angle * 180) / np.pi
    steering.append(angle)

    # left image append
    img_path = os.path.join(datadir,left.strip())
    img = load_img(img_path,shape)
    images.append(img)
    angle = float(indexed_data[3])+0.15
    angle = (angle * 180) / np.pi
    steering.append(angle)

    # right image append
    img_path = os.path.join(datadir,right.strip())
    img = load_img(img_path,shape)
    images.append(img)
    angle = float(indexed_data[3])-0.15
    angle = (angle * 180) / np.pi
    steering.append(angle)


  images      = np.stack(images)
  steerings   = np.asarray(steering)
  return images, steerings
 



def load_data(shape):
  image_paths, steerings = load_img_steering(f'{datadir}/car_simulator_dataset/IMG',shape=shape)

  x_train, x_val, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

  print(f"DATASET SHAPE: TRAIN: [{x_train.shape}] , [{y_train.shape}]")
  return x_train, x_val, y_train, y_val

def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.3))
  image = zoom.augment_image(image)
  return image


def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle


def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)
    
    return image, steering_angle

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def batch_generator(image_paths, steering_ang, batch_size, istraining):
  
  while True:
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)
      
      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      
      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering)) 

def nvidiaModel():
  model = Sequential()
  model.add(Convolution2D(24,(5,5),strides=(2,2),input_shape=(66,200,3),activation="elu"))
  model.add(Convolution2D(36,(5,5),strides=(2,2),activation="elu"))
  model.add(Convolution2D(48,(5,5),strides=(2,2),activation="elu")) 
  model.add(Convolution2D(64,(3,3),activation="elu"))   
  model.add(Convolution2D(64,(3,3),activation="elu"))
  model.add(Dropout(0.5))
  
  model.add(GlobalAveragePooling2D())
  
  model.add(Dense(128,activation="elu"))
  model.add(Dropout(0.5))
  
  model.add(Dense(64,activation="elu"))
  model.add(Dropout(0.5))
  
  model.add(Dense(16,activation="elu"))
  model.add(Dropout(0.5))
  
  model.add(Dense(1))
  model.compile(loss='mean_sqared_error',optimizer=Adam(lr=1e-3),metrics=['accuracy','mean_absolute_error'])
  
  return model

