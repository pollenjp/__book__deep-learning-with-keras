import os
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist


def lenet(input_shape, num_classes):
    model = Sequential()

    # extract image features by convolution and max pooling layers
    model.add( Conv2D( 20, kernel_size=5, padding="same", input_shape=input_shape, activation="relu" ) )
    model.add( MaxPooling2D(pool_size=(2, 2)) )
    model.add( Conv2D( 50, kernel_size=5, padding="same", activation="relu") )
    model.add( MaxPooling2D(pool_size=(2, 2)) )
    # classify the class by fully-connected layers
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model


