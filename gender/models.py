from keras.layers import Activation, Convolution2D, Dropout, Dense, Flatten
from keras.layers import AveragePooling2D, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from spatial_transformer import SpatialTransformer
import numpy as np

def simple_CNN(input_shape, num_classes):

    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(5, 5),strides=(2, 2),
                                                padding='same'))
    model.add(Dropout(.5))
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2),
                                                padding='same'))
    model.add(Dropout(.5))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2),
                                                padding='same'))
    model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dense(1028))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1028))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax',name='predictions'))
    return model

def attention_CNN(input_shape, num_classes):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]

    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
    locnet.add(Convolution2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Convolution2D(20, (5, 5)))
    locnet.add(Flatten())
    locnet.add(Dense(50))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights))

    model = Sequential()
    model.add(SpatialTransformer(localization_net=locnet,
                                 output_size=(30,30),
                                 input_shape=input_shape,
                                 name='image_array'))
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                                                input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(5, 5),strides=(2, 2),
                                                padding='same'))
    model.add(Dropout(.5))
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2),
                                                padding='same'))
    model.add(Dropout(.5))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2),
                                                padding='same'))
    model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dense(1028))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1028))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax', name='predictions'))
    return model

