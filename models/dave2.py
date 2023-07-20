from keras import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Dense

#our model is Nvidia Dave-2 where you can find here: https://arxiv.org/pdf/1604.07316.pdf
def Dave2():
    model = Sequential()

    # 5x5 Convolutional layers with stride of 2x2
    model.add(Conv2D(24, (5, 5), strides=(2, 2),activation='elu',input_shape=input_shape))
    model.add(Conv2D(36, (5, 5), strides=(2, 2),activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2),activation='elu'))
    
    # 3x3 Convolutional layers with stride of 1x1
    model.add(Conv2D(64, (3, 3),activation='elu'))
    model.add(Conv2D(64, (3, 3),activation='elu'))
    
    # Flatten before passing to the fully connected layers
    model.add(Flatten())
    # Three fully connected layers
    model.add(Dense(100,activation='elu'))
    model.add(Dropout(.25))
    model.add(Dense(50,activation='elu'))
    model.add(Dropout(.25))
    model.add(Dense(10,activation='elu'))
    model.add(Dropout(.25))
    
    # Output layer with linear activation 
    model.add(Dense(1,activation="linear"))
    
    return model