## Making essential imports
import os
import numpy as np
import pandas as pd
import re
import matplotlib

import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm


import pathlib
directory = str(pathlib.Path(__file__).parent.absolute()) + "/CamVid/"
import pandas as pd 


#LOAD CLASSES
classes = pd.read_csv(directory +'class_dict.csv', index_col =0)
#LOAD CONVERSIONS DICTIONARY
cls2rgb = {cl:list(classes.loc[cl, :]) for cl in classes.index}
idx2rgb={idx:np.array(rgb) for idx, (cl, rgb) in enumerate(cls2rgb.items())}


def adjust_mask(mask,class_mapping= cls2rgb):
    
    semantic_map = []
    for colour in list(class_mapping.values()):        
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return np.float32(semantic_map)

def map_rgb_to_class(masks,cls2rgb=cls2rgb,verbose=True):

    new_masks = []
    for mask in tqdm(masks,disable=not verbose):
        m = adjust_mask(mask,cls2rgb)
        new_masks.append(m)

    return np.stack(new_masks)



def map_class_to_rgb(p):
    

    idx = np.argmax(p)
    rgb = idx2rgb[idx]

    #print(f'CLASS[{idx}] => [{rgb}] ')
    return idx2rgb[idx]

from tensorflow.keras.preprocessing.image import ImageDataGenerator
def get_generator(shape=256,batch_size = 32,not_train=False,not_valid=False):

    val_generator_fn    = None
    train_generator_fn  = None

    if not not_valid:
        #for validation data:
        image_datagen = ImageDataGenerator(rescale=1./255)
        mask_datagen  = ImageDataGenerator()
        val_image_generator = image_datagen.flow_from_directory(
            directory,
            class_mode=None,
            classes=['val'],
            seed=444,
            batch_size=batch_size,
            target_size=(shape,shape))

        val_mask_generator = mask_datagen.flow_from_directory(
            directory,
            classes=['val_labels'],
            class_mode=None,
            seed=444,
            batch_size=batch_size,
            color_mode='rgb',
            target_size=(shape,shape))

        # combine generators into one which yields image and masks
        val_generator = zip(val_image_generator, val_mask_generator)

        def val_generator_fn():
            for (img,mask) in val_generator:
                new_mask = adjust_mask(mask)
                yield (img,new_mask) 
        #for training data:

    if not not_train:
        train_image_datagen = ImageDataGenerator(rotation_range=10,
            width_shift_range=0.2,
            zoom_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            rescale=1./255)
        train_mask_datagen = ImageDataGenerator(rotation_range=10,
            width_shift_range=0.2,
            zoom_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        image_generator =train_image_datagen.flow_from_directory(
            directory,
            class_mode=None,
            classes=['train'],
            seed=444,
            batch_size=batch_size,
            target_size=(shape,shape))

        mask_generator =train_mask_datagen.flow_from_directory(
            directory,
            classes=['train_labels'],
            class_mode=None,
            seed=444,
            color_mode='rgb',
            batch_size=batch_size,
            target_size=(shape,shape))
        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, mask_generator)

        def train_generator_fn():
            for (img,mask) in train_generator:
                new_mask = adjust_mask(mask)
                yield (img,new_mask)  

    return train_generator_fn, val_generator_fn

def loadDataWithClassDict(shape=256):
    x_train,y_train,x_val,y_val = loadData(shape)

    y_train = map_rgb_to_class(y_train)
    y_val   = map_rgb_to_class(y_val)

    return x_train,y_train,x_val,y_val


## defining data Loader function
def loadData(shape = 256):

    img_train_path  = directory + "train/"
    mask_train_path = directory + "train_labels/"
    img_valid_path  = directory + "val/"
    mask_valid_path = directory + "val_labels/"
    
    frameObj_train= {'img' : [],'mask' : []}
    frameObj_valid= {'img' : [],'mask' : []}

    train_img_names  = os.listdir(img_train_path)
    valid_img_names  = os.listdir(img_valid_path)

    train_mask_names = []
    valid_mask_names = []

    ## generating mask names
    for mem in tqdm(train_img_names):
        train_mask_names.append(re.sub('\.png', '_L.png', mem))
    for mem in tqdm(valid_img_names):
        valid_mask_names.append(re.sub('\.png', '_L.png', mem))

    for i in tqdm(range (len(train_img_names))):
        img_train   = plt.imread(img_train_path     + train_img_names[i])
        mask_train  = plt.imread(mask_train_path    + train_mask_names[i])

        img = cv2.resize(img_train, (shape, shape))
        mask = cv2.resize(mask_train, (shape, shape))

        frameObj_train['img'].append(img)
        frameObj_train['mask'].append(mask)

    for i in tqdm(range (len(valid_img_names))):
        img_valid   = plt.imread(img_valid_path     + valid_img_names[i])
        mask_valid  = plt.imread(mask_valid_path    + valid_mask_names[i])

        img = cv2.resize(img_valid, (shape, shape))
        mask = cv2.resize(mask_valid, (shape, shape))

        frameObj_valid['img'].append(img)
        frameObj_valid['mask'].append(mask)

    return np.array(frameObj_train['img']),np.array(frameObj_train['mask']),np.array(frameObj_valid['img']),np.array(frameObj_valid['mask'])





## function for getting 16 predictions
def predict (imgs,masks, model, shape = 256):
    ## getting and proccessing val data


    predictions = model.predict(imgs,verbose=False)
    for i in range(len(imgs)):
        predictions[i] = cv2.merge((predictions[i,:,:,0],predictions[i,:,:,1],predictions[i,:,:,2]))

    return predictions, imgs, masks




def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(7,7))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('image')

    plt.subplot(1,3,2)
    plt.imshow(predMask)
    plt.title('Predicted Mask')

    plt.subplot(1,3,3)
    plt.imshow(groundTruth)
    plt.title('actual Mask')

    plt.show()