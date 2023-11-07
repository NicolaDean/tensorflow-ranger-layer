import numpy as np

from keras import datasets, layers, models, losses
from keras.utils import img_to_array, array_to_img
VALIDATION_SIZE = 100

def load_data(shape = (48,48)):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = np.dstack([x_train] * 3)
    x_test = np.dstack([x_test] * 3)

    x_train = x_train.reshape(-1, 28,28,3)
    x_test = x_test.reshape(-1,28,28,3)

    x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize(shape)) for im in x_train])
    x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize(shape)) for im in x_test])

    x_val = x_train[-VALIDATION_SIZE:, :, :, :]
    y_val = y_train[-VALIDATION_SIZE:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]

    return x_train, y_train, x_val, y_val