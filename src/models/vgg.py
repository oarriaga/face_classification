from keras.layers import Activation, Conv2D
from keras.layers import BatchNormalization
from keras.layers import MaxPool2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model


def build_vgg(input_shape, num_classes, stem_kernels, block_kernels):

    x = inputs = Input(input_shape, name='inputs')
    for num_kernels in stem_kernels:
        x = Conv2D(num_kernels, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    for num_kernels in block_kernels:
        x = Conv2D(num_kernels, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(2)(x)

    x = Conv2D(num_classes, 3, padding='same')(x)
    # x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    model_name = '-'.join(['VGG',
                           str(input_shape[0]),
                           str(stem_kernels[0]),
                           str(len(block_kernels))
                           ])
    model = Model(inputs, output, name=model_name)
    return model
