from keras.layers import Activation, Conv2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input
from keras.regularizers import l2
from .blocks import xception_block


def build_xception(input_shape, num_classes, stem_kernels,
                   block_kernels, l2_reg=0.01):

    x = inputs = Input(input_shape, name='inputs')
    for num_kernels in stem_kernels:
        x = Conv2D(num_kernels, 3, kernel_regularizer=l2(l2_reg),
                   use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    for num_kernels in block_kernels:
        x = xception_block(x, num_kernels, l2_reg)

    x = Conv2D(num_classes, 3, kernel_regularizer=l2(l2_reg),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model_name = '-'.join(['XCEPTION',
                           str(input_shape[0]),
                           str(stem_kernels[0]),
                           str(len(block_kernels))
                           ])
    model = Model(inputs, output, name=model_name)
    return model
