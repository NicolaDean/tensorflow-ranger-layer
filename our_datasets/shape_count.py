import pandas as pd
import pathlib
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import pickle
from sklearn.model_selection import train_test_split
from keras import backend as K
IMG_HEIGHT = 128
IMG_WIDTH  = 128


directory = str(pathlib.Path(__file__).parent.absolute())

data_dir    = directory + '/shape_count'
train_path  = directory + '/shape_count/train_images'
test_path   = directory + '/shape_count/test_images'
train_label = directory + '/shape_count/train_labels.csv'
test_label  = directory + '/shape_count/test_labels.csv'


def load_dataset(shape = (IMG_HEIGHT, IMG_WIDTH)):
    with open(f'{data_dir}/train_images_transparent.pkl', 'rb') as inputfile:
        x_train = pickle.load(inputfile)
    
    '''
    with open(f'{data_dir}/test_images_transparent.pkl', 'rb') as inputfile:
        x_test = pickle.load(inputfile)
    '''
    y_train = pd.read_csv(train_label)
    #y_test  = pd.read_csv(test_label,header = None)

    y_train = np.array(y_train)
    #y_test  = np.array(y_test)

    img_rows, img_cols = 100,100

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        #x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,3)
        #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,3)
        input_shape = (img_rows, img_cols, 3)
    print(input_shape)
    print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')
    
    x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42, shuffle=True)


    return x_train, x_val, y_train, y_val

def load_dataset_1(shape = (IMG_HEIGHT, IMG_WIDTH)):
    labels = pd.read_csv(train_label)

    labels = labels.reset_index()  # make sure indexes pair with number of rows

    print(labels.columns)
    image_data      = []
    image_labels    = []
    #for index, row in tqdm(labels.iterrows(),total=labels.shape[0]):
    for id in tqdm(labels.index):
        img_name    = f'{train_path}/{id+1}.png'
        label       = np.array((labels["b"][id],labels["g"][id],labels["r"][id]))
        
        image           =  cv2.imread(img_name)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image    = image_fromarray.resize(shape)
        image_data.append(np.array(resize_image))
        image_labels.append(label) 

     # Changing the list to numpy array
    image_data = np.array(image_data)
    image_labels = np.array(image_labels)

    print(image_data.shape, image_labels.shape)

    x_train, x_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

    return x_train, x_val, y_train, y_val
