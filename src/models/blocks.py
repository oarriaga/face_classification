from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import add
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import Reshape, Lambda, dot
from keras.layers import MaxPool1D
from keras import backend as K


from keras.regularizers import l2


def dense_block(x, blocks, growth_rate, name):
    """DenseNet core block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x


def conv_block(x, growth_rate, name):
    """Standard convolution block used in DenseNets.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis,
                            epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                padding='same',
                use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def transition_block(x, reduction, name):
    """Transition block used for DenseNets.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
               use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def xception_block(input_tensor, num_kernels, l2_reg=0.01):
    """Xception core block.
    # Arguments
        input_tenso: Keras tensor.
        num_kernels: Int. Number of convolutional kernels in block.
        l2_reg: Float. l2 regression.
    # Returns
        output tensor for the block.
    """
    residual = Conv2D(num_kernels, 1, strides=(2, 2),
                      padding='same', use_bias=False)(input_tensor)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        num_kernels, 3, padding='same',
        kernel_regularizer=l2(l2_reg), use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(num_kernels, 3, padding='same',
                        kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=(2, 2), padding='same')(x)
    x = add([x, residual])
    return x


def non_local_block(input_tensor, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    """Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial)
    or 5 (spatio-temporal).
    Implementation modified from the following reference:
    https://github.com/titu1994/keras-non-local-nets/blob/master/non_local.py

    # Arguments:
        input_tensor: Keras tensor
        intermediate_dim: Positive Integer.
            Dimension of the intermediate representation.
            If `None`, computes the intermediate dimension as half of the
            input channel dimension.
        compression: Positive integer. Compresses the intermediate
            representation during the dot products to reduce memory usage.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step.
            Set to 1 to prevent computation compression.
            None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot`.
        add_residual: Boolean value to decide if the residual
            connection should be added or not.
            Default is True for ResNets, and False for Self Attention.

    # Returns:
        Keras tensor. Returned tensor has same shape as input
    """

    batch_size, height, width, num_channels = K.int_shape(input_tensor)
    tensor_rank = len(K.int_shape(input_tensor))

    if tensor_rank != 4:
        raise ValueError('Input tensor has to have shape of 4')

    if mode not in ['gaussian', 'embedded', 'dot']:
        raise ValueError('"mode" must be: "gaussian", "embedded", "dot"')

    if intermediate_dim is None:
        intermediate_dim = int(num_channels / 2)

    if mode == 'gaussian':
        x_i = Reshape((-1, num_channels))(input_tensor)
        x_j = Reshape((-1, num_channels))(input_tensor)

        f = dot([x_i, x_j], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':
        theta = channel_conv2D(input_tensor, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        phi = channel_conv2D(input_tensor, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)
        # scale the values to make it size invariant
        size = float(K.int_shape(f)[-1])
        f = Lambda(lambda x: x / size)(f)

    elif mode == 'embedded':
        theta = channel_conv2D(input_tensor, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        phi = channel_conv2D(input_tensor, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)
        if compression > 1:
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    g = channel_conv2D(input_tensor, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        g = MaxPool1D(compression)(g)

    y = dot([f, g], axes=[2, 1])
    y = Reshape((height, width, intermediate_dim))(y)
    y = channel_conv2D(y, num_channels)
    if add_residual:
        y = add([input_tensor, y])
    return y


def channel_conv2D(input_tensor, num_channels):
    return Conv2D(num_channels, (1, 1), padding='same', use_bias=False,
                  kernel_initializer='he_normal')(input_tensor)
