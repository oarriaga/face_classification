from keras.layers import Activation, Conv2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model


def build_vgg(input_shape, num_classes, kernels_per_block):

    inputs = Input(input_shape, name='inputs')
    x = Conv2D(32, 3, padding='same')(inputs)
    x = BatchNormalization()
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()
    x = Activation('relu')(x)

    for num_kernels in kernels_per_block:
        x = Conv2D(x, num_kernels)
        x = BatchNormalization()
        x = Activation('relu')(x)

    x = Conv2D(num_classes, 3, padding='same')(x)
    x = BatchNormalization()
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    model_name = ''.join(['VGG', str(len(kernels_per_block))])
    model = Model(inputs, output, name=model_name)
    return model
