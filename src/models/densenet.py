import os

from keras import layers
from keras import backend
from keras import models
from keras.layers import Input

from .blocks import dense_block
from .blocks import transition_block


def build_densenet(input_shape, num_classes, kernels_per_block):
    """Instantiates the DenseNet architecture.
    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False. It should have 3 inputs channels.
        num_classes: optional number of num_classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        kernels_per_block: numbers of kernels per dense blocks.
    # Returns
        A Keras model instance.
    """

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    inputs = Input(input_shape, name='inputs')

    x = layers.Conv2D(32, 3, use_bias=False, padding='same',
                      name='conv1a/conv')(inputs)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1a/bn')(x)
    x = layers.Activation('relu', name='conv1a/relu')(x)

    x = layers.Conv2D(64, 3, use_bias=False, padding='same',
                      name='conv1b/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1b/bn')(x)
    x = layers.Activation('relu', name='conv1b/relu')(x)

    for blocks_arg, num_blocks in enumerate(kernels_per_block[:-1], 1):
        x = dense_block(x, num_blocks, name='conv%i' % blocks_arg)
        x = transition_block(x, 0.5, name='pool%i' % blocks_arg)

    x = dense_block(x, kernels_per_block[-1],
                    name='conv%i' % len(kernels_per_block))
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, activation='softmax', name='fc1000')(x)
    return models.Model(inputs, x, name='DenseNet')
