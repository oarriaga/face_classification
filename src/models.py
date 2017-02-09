from keras.layers import Activation, Convolution2D, Dropout, Dense, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import AveragePooling2D, BatchNormalization
from keras.models import Sequential

def simple_CNN(input_shape, num_classes):

    model = Sequential()

    model.add(Convolution2D(16, 7, 7, border_mode='same',
                            input_shape=input_shape))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(5, 5),strides=(2, 2), border_mode='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2), border_mode='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2), border_mode='same'))
    model.add(Dropout(.5))

    model.add(Flatten())
    model.add(Dense(1028))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1028))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
