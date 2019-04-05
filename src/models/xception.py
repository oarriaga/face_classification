from keras.layers import Activation, Conv2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input
from keras.regularizers import l2
from .blocks import xception_block


def build_xception(input_shape, num_classes,
                   kernels_per_module, l2_reg=0.01):

    inputs = Input(input_shape)
    x = Conv2D(32, 3, kernel_regularizer=l2(l2_reg), use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for num_kernels in kernels_per_module:
        x = xception_block(x, num_kernels, l2_reg)

    x = Conv2D(num_classes, 3, kernel_regularizer=l2(l2_reg),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model_name = ''.join(['Xception', str(len(kernels_per_module))])
    model = Model(inputs, output, name=model_name)
    return model
