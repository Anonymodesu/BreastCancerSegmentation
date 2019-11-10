"""A vanilla 3D resnet implementation. From https://github.com/JihongJu/keras-resnet3d

Based on Raghavendra Kotikalapudi's 2D implementation
keras-resnet (See https://github.com/raghakot/keras-resnet.)

removed initial subsampling for segmentation
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Lambda,
    concatenate
)
from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D,
    Conv3DTranspose
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import math

def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(deconv=False, **conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        if deconv:
            conv = Conv3DTranspose(filters=filters, kernel_size=kernel_size,
                          strides=strides, kernel_initializer=kernel_initializer,
                          padding=padding,
                          kernel_regularizer=kernel_regularizer)(input)
        else:
            conv = Conv3D(filters=filters, kernel_size=kernel_size,
                          strides=strides, kernel_initializer=kernel_initializer,
                          padding=padding,
                          kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv3d(deconv=False, **conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        
        if deconv:
            return Conv3DTranspose(filters=filters, kernel_size=kernel_size,
                          strides=strides, kernel_initializer=kernel_initializer,
                          padding=padding,
                          kernel_regularizer=kernel_regularizer)(activation)            
        else:
            return Conv3D(filters=filters, kernel_size=kernel_size,
                          strides=strides, kernel_initializer=kernel_initializer,
                          padding=padding,
                          kernel_regularizer=kernel_regularizer)(activation)
    return f

#if deconv=False, input is larger than residual
#if deconv=True, input is smaller than residual
def _shortcut3d(input, residual, deconv=False):
    """3D shortcut to match input and residual and merges them with "sum"."""
    if deconv:
        stride_dim1 = math.ceil((residual._keras_shape[DIM1_AXIS] / input._keras_shape[DIM1_AXIS]))
        stride_dim2 = math.ceil((residual._keras_shape[DIM2_AXIS] / input._keras_shape[DIM2_AXIS]))
        stride_dim3 = math.ceil((residual._keras_shape[DIM3_AXIS] / input._keras_shape[DIM3_AXIS]))
    else:
        stride_dim1 = math.ceil((input._keras_shape[DIM1_AXIS] / residual._keras_shape[DIM1_AXIS]))
        stride_dim2 = math.ceil((input._keras_shape[DIM2_AXIS] / residual._keras_shape[DIM2_AXIS]))
        stride_dim3 = math.ceil((input._keras_shape[DIM3_AXIS] / residual._keras_shape[DIM3_AXIS]))
        
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        
        if deconv:
            shortcut = Conv3DTranspose(
                filters=residual._keras_shape[CHANNEL_AXIS],
                kernel_size=(1, 1, 1),
                strides=(stride_dim1, stride_dim2, stride_dim3),
                kernel_initializer="he_normal", padding="valid",
                kernel_regularizer=l2(1e-4)
                )(input)
            
        else:
            shortcut = Conv3D(
                filters=residual._keras_shape[CHANNEL_AXIS],
                kernel_size=(1, 1, 1),
                strides=(stride_dim1, stride_dim2, stride_dim3),
                kernel_initializer="he_normal", padding="valid",
                kernel_regularizer=l2(1e-4)
                )(input)
            
    return add([shortcut, residual])


def _residual_block3d(block_function, filters, kernel_regularizer, repetitions,
                      is_first_layer=False, deconv=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = block_function(deconv=deconv, filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0)
                                   )(input)
        return input

    return f


def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False, deconv=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            if deconv:
                conv1 = Conv3DTranspose(filters=filters, kernel_size=(3, 3, 3),
                               strides=strides, padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=kernel_regularizer
                               )(input)
            else:
                conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                               strides=strides, padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=kernel_regularizer
                               )(input)
        else:
            conv1 = _bn_relu_conv3d(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides,
                                    kernel_regularizer=kernel_regularizer
                                    )(input)

        residual = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv1)
        return _shortcut3d(input, residual)

    return f


def bottleneck(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
               is_first_block_of_first_layer=False, deconv=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            if deconv:
                conv_1_1 = Conv3DTranspose(filters=filters, kernel_size=(1, 1, 1),
                                  strides=strides, padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=kernel_regularizer
                                  )(input)                
            else:
                conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                                  strides=strides, padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=kernel_regularizer
                                  )(input)
        else:
            conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
                                       strides=strides,
                                       kernel_regularizer=kernel_regularizer
                                       )(input)

        conv_3_3 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_1_1)
        residual = _bn_relu_conv3d(filters=filters * 4, kernel_size=(1, 1, 1),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_3_3)

        return _shortcut3d(input, residual)

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

class Resnet3DBuilder(object):
    """ResNet3D."""

    @staticmethod
    def build(input_shape, output_shape, block_fn, repetitions, reg_factor, mode='detection'):
                    
        """Instantiate a vanilla ResNet3D keras model.

        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            output_shape: Tuple of output shape
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
            mode: detection or segmentation
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
    
        _handle_data_format()
        
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")
            
        input = Input(shape=input_shape)
  
        if mode == 'localisation':
            subsampling_strides = (2, 2, 2)
        
        elif mode == 'segmentation':
            subsampling_strides = (1, 1, 1)
            
        
        block_fn = _get_block(block_fn)
        # first conv
        conv1 = _conv_bn_relu3D(filters=64, kernel_size=(7, 7, 7),
                                strides=subsampling_strides,
                                kernel_regularizer=l2(reg_factor)
                                )(input)
        
        if mode == 'localisation':
            conv1 = MaxPooling3D(pool_size=(3, 3, 3), strides=subsampling_strides,
                             padding="same")(conv1)

        # repeat blocks
        block = conv1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block3d(block_fn, filters=filters,
                                      kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            filters *= 2

        print
            
        # last activation
        block_output = _bn_relu(block)
        
        print(block_output)
        

        if mode == 'detection':
            # average poll and classification
            pool2 = AveragePooling3D(pool_size=(block._keras_shape[DIM1_AXIS],
                                        block._keras_shape[DIM2_AXIS],
                                        block._keras_shape[DIM3_AXIS]),
                             strides=(1, 1, 1))(block_output)
            
            flatten1 = Flatten()(pool2) 
            
            dense_centre = Dense(units=output_shape[0],
                          kernel_initializer="he_normal",
                          activation='linear',
                          kernel_regularizer=l2(reg_factor))(flatten1)
            
            model = Model(inputs=input, outputs=dense_centre)
            
        elif mode == 'localisation':
            # average poll and classification
            pool2 = AveragePooling3D(pool_size=(block._keras_shape[DIM1_AXIS],
                                        block._keras_shape[DIM2_AXIS],
                                        block._keras_shape[DIM3_AXIS]),
                             strides=(1, 1, 1))(block_output)
            
            flatten1 = Flatten()(pool2) 
            
            #centre prediction
            dense_centre = Dense(units=output_shape[0],
                          kernel_initializer="he_normal",
                          activation='linear',
                          kernel_regularizer=l2(reg_factor))(flatten1)
            
            #size prediction
            dense_size = Dense(units=output_shape[1],
                          kernel_initializer="he_normal",
                          activation='relu', #size has to be positive
                          kernel_regularizer=l2(reg_factor))(flatten1)
            
            #transform centre/size from (?, 3) to (?, 1, 3) tensors so that they could be concatenated to (?, 2, 3) tensors
            dense_centre = Lambda(lambda x: K.expand_dims(x, axis=1))(dense_centre)
            dense_size = Lambda(lambda x: K.expand_dims(x, axis=1))(dense_size)
            output = concatenate([dense_centre, dense_size], axis=1)
            
            model = Model(inputs=input, outputs=output)
        
        elif mode == 'segmentation':
            # deconvolution time o_O; 
            # approximately the same layer structure as the convolution half but in reverse
            
            
            deconv = Conv3DTranspose(filters=64, kernel_size=(4,4,5),
                          strides=(1,1,1), kernel_initializer='he_normal',
                          kernel_regularizer=l2(reg_factor))(block_output)
            deconv = _bn_relu(deconv)
            
            deconv = Conv3DTranspose(filters=64, kernel_size=(4,4,5),
                          strides=(1,1,1), kernel_initializer='he_normal',
                          kernel_regularizer=l2(reg_factor))(deconv)
            deconv = _bn_relu(deconv)
            
            deconv_block_output = _shortcut3d(block_output, deconv, deconv=True)
            
            
            deconv = Conv3DTranspose(filters=64, kernel_size=(5,5,7),
                          strides=(1,1,1), kernel_initializer='he_normal',
                          kernel_regularizer=l2(reg_factor))(deconv_block_output)
            deconv = _bn_relu(deconv)
            
            deconv = Conv3DTranspose(filters=64, kernel_size=(6,6,7),
                          strides=(1,1,1), kernel_initializer='he_normal',
                          kernel_regularizer=l2(reg_factor))(deconv)
            deconv = _bn_relu(deconv)
            
            deconv_block_output = _shortcut3d(deconv_block_output, deconv, deconv=True)

            
            deconv = Conv3DTranspose(filters=64, kernel_size=(6,7,9),
                          strides=(1,1,1), kernel_initializer='he_normal',
                          kernel_regularizer=l2(reg_factor))(deconv)#deconv_block_output)
            deconv = _bn_relu(deconv)

            deconv = Conv3D(filters=1, kernel_size=(1,1,1),
                          strides=(1,1,1), kernel_initializer='he_normal',
                          kernel_regularizer=l2(reg_factor))(deconv)
            deconv = Lambda(lambda x: K.squeeze(x, -1))(deconv)

            final_activation = Activation('sigmoid', name='final_activation')(deconv)

            model = Model(inputs=input, outputs=[final_activation])

        else:
            model = None
            print('Wrong mode selected:', mode)

        
        return model

    @staticmethod
    def build_resnet_18(input_shape, output_shape, reg_factor=1e-4, mode='detection'):
        """Build resnet 18."""
        return Resnet3DBuilder.build(input_shape, output_shape, basic_block,
                                     [2, 2, 2, 2], reg_factor=reg_factor, mode=mode)

    @staticmethod
    def build_resnet_34(input_shape, output_shape, reg_factor=1e-4, mode='detection'):
        """Build resnet 34."""
        return Resnet3DBuilder.build(input_shape, output_shape, basic_block,
                                     [3, 4, 6, 3], reg_factor=reg_factor, mode=mode)

    @staticmethod
    def build_resnet_50(input_shape, output_shape, reg_factor=1e-4, mode='detection'):
        """Build resnet 50."""
        return Resnet3DBuilder.build(input_shape, output_shape, bottleneck,
                                     [3, 4, 6, 3], reg_factor=reg_factor, mode=mode)

    @staticmethod
    def build_resnet_101(input_shape, output_shape, reg_factor=1e-4, mode='detection'):
        """Build resnet 101."""
        return Resnet3DBuilder.build(input_shape, output_shape, bottleneck,
                                     [3, 4, 23, 3], reg_factor=reg_factor, mode=mode)

    @staticmethod
    def build_resnet_152(input_shape, output_shape, reg_factor=1e-4, mode='detection'):
        """Build resnet 152."""
        return Resnet3DBuilder.build(input_shape, output_shape, bottleneck,
                                     [3, 8, 36, 3], reg_factor=reg_factor, mode=mode)