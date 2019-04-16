from keras import layers
from keras import backend
from keras import models
from keras.layers import Input

from .blocks import dense_block
from .blocks import transition_block


def build_densenet(input_shape, num_classes, stem_kernels,
                   blocks, growth_rate=32):
    """Instantiates the DenseNet architecture.
    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False. It should have 3 inputs channels.
        num_classes: optional number of num_classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        block_kernels: numbers of kernels per dense blocks.
    # Returns
        A Keras model instance.
    """

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = inputs = Input(input_shape, name='inputs')
    for stem_arg, num_kernels in enumerate(stem_kernels):
        x = layers.Conv2D(stem_kernels[0], 3, use_bias=False, padding='same',
                          name='conv/conv%i' % stem_arg)(inputs)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv/bn%i' % stem_arg)(x)
        x = layers.Activation('relu', name='conv/relu%i' % stem_arg)(x)

    for block_arg, num_blocks in enumerate(blocks[:-1]):
        x = dense_block(x, num_blocks, growth_rate, name='conv%i' % block_arg)
        x = transition_block(x, 0.5, name='pool%i' % block_arg)

    name = 'conv%i' % len(blocks)
    x = dense_block(x, blocks[-1], growth_rate, name=name)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, activation='softmax', name='fc1000')(x)
    return models.Model(inputs, x, name='DenseNet')
