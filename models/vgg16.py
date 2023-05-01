from tensorflow import keras
from keras import Sequential, layers
import tensorflow as tf

def VGG16(input_shape):
    input = layers.Input(shape = input_shape)
    
    #1st conv block
    x = layers.Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
    x = layers.Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    #2nd conv block
    x = layers.Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    #3rd conv block
    x = layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    #4th conv block
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    #5th conv block 
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    #final dense 
    x = layers.Flatten()(x) 
    x = layers.Dense(units = 500, activation = 'relu')(x)
    output = layers.Dense(units = 10, activation ='softmax')(x)
    model = keras.Model(inputs=input, outputs =output)
    model.build(input_shape)
    return model