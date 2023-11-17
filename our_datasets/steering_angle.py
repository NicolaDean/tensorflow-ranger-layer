import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_data(shape):
    with open("../our_datasets/train_steering_wheel/data.txt") as f:
        lines = f.readlines()
    
    lines = [[x.split()[0], x.split()[1]] for x in lines]
    print(lines[1])

    image_data = []
    image_labels = []
    for row in tqdm(lines[:3000]):
        path = "../our_datasets/train_steering_wheel/"+row[0]
        image_labels.append(float(row[1]))
        image = cv2.imread(path)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize(shape)
        image_data.append(np.array(resize_image))

    x_train, x_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

    x_train = np.array(x_train, dtype = float)
    x_val = np.array(x_val, dtype = float)
    y_train = np.array(y_train, dtype = float)
    y_val = np.array(y_val, dtype = float)    
    return x_train, x_val, y_train, y_val
