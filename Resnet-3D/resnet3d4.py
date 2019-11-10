"""A vanilla 3D resnet implementation. From https://github.com/JihongJu/keras-resnet3d

Based on Raghavendra Kotikalapudi's 2D implementation
keras-resnet (See https://github.com/raghakot/keras-resnet.)

Refinement no longer predicts size
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
    concatenate,
    Multiply,
    Permute,
    Add
)
from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D,
    Conv3DTranspose,
    ZeroPadding3D,
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import math
import tensorflow as tf
from tfscripts.conv import conv4d_stacked
from conv4d import conv4d



def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    batch_norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(batch_norm)


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
        stride_dim1 = math.floor(residual._keras_shape[DIM1_AXIS] / input._keras_shape[DIM1_AXIS])
        stride_dim2 = math.floor(residual._keras_shape[DIM2_AXIS] / input._keras_shape[DIM2_AXIS])
        stride_dim3 = math.floor(residual._keras_shape[DIM3_AXIS] / input._keras_shape[DIM3_AXIS])
        
        padding_dim1 = (residual._keras_shape[DIM1_AXIS] ) % input._keras_shape[DIM1_AXIS]
        padding_dim2 = (residual._keras_shape[DIM2_AXIS] ) % input._keras_shape[DIM2_AXIS]
        padding_dim3 = (residual._keras_shape[DIM3_AXIS]) % input._keras_shape[DIM3_AXIS]
        padding_dim1 = (math.floor(0.5 * padding_dim1), math.ceil(0.5 * padding_dim1))
        padding_dim2 = (math.floor(0.5 * padding_dim2), math.ceil(0.5 * padding_dim2))
        padding_dim3 = (math.floor(0.5 * padding_dim3), math.ceil(0.5 * padding_dim3))

    else:
        stride_dim1 = math.ceil(input._keras_shape[DIM1_AXIS] / residual._keras_shape[DIM1_AXIS])
        stride_dim2 = math.ceil(input._keras_shape[DIM2_AXIS] / residual._keras_shape[DIM2_AXIS])
        stride_dim3 = math.ceil(input._keras_shape[DIM3_AXIS] / residual._keras_shape[DIM3_AXIS])
        
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        
        if deconv: #output = stride_dim * input_dim + padding_dim
            shortcut = Conv3DTranspose(
                filters=residual._keras_shape[CHANNEL_AXIS],
                kernel_size=(1, 1, 1),
                strides=(stride_dim1, stride_dim2, stride_dim3),
                kernel_initializer="he_normal", padding="valid",
                kernel_regularizer=l2(1e-4)
                )(input)
            
            shortcut = ZeroPadding3D(padding=(padding_dim1, padding_dim2, padding_dim3))(shortcut)
                        
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
            if i == 0 and not is_first_layer and not deconv:
                strides = (2, 2, 2)
            input = block_function(deconv=deconv, filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0),
                                   upsample=(deconv and i == 0) #only up/downsample the first layer of each block
                                   )(input)
        return input

    return f


def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False, deconv=False, upsample=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        
        
        if deconv: #deconvolute by upsampling and convolution
            
            if upsample:
                input = upsample3d(2)(input)
                
            conv1 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
               strides=strides, padding="same",
               kernel_initializer="he_normal",
               kernel_regularizer=kernel_regularizer,
               )(input)
            residual = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
               strides=strides, padding="same",
               kernel_initializer="he_normal",
               kernel_regularizer=kernel_regularizer,
               )(conv1)
            
            #print(residual)
        
        else:
            
            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
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

        if Resnet3DBuilder.use_shortcut:
            return _shortcut3d(input, residual, deconv)
        else:
            return residual
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

#from https://github.com/keras-team/keras/issues/890
#usage: x = crop(2,5,10)(x)
def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def expand_dims(axis):
    return Lambda(lambda x: K.expand_dims(x, axis=axis))
    
def permute(pattern):
    return Lambda(lambda x: K.permute_dimensions(x, pattern))

def squeeze(axis):
    return Lambda(lambda x: K.squeeze(x, axis))

def conv4d_wrapper(num_filters, kernel_size, reg_factor, padding, index):
    return Lambda(lambda x: conv4d(input=x, #conv4d only accepts channels_first format
                                  filters=num_filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  activation=None,
                                  kernel_regularizer=l2(reg_factor),
                                  kernel_initializer="he_normal",
                                  data_format='channels_first',
                                 name='conv4d{}_{}'.format(Resnet3DBuilder.iteration, index),
                                 reuse=tf.AUTO_REUSE))

def upsample3d_helper(x,size):
    x = K.repeat_elements(x, size, 1)
    x = K.repeat_elements(x, size, 2)
    x = K.repeat_elements(x, size, 3)
    return x 

def upsample3d(size):
    return Lambda(lambda x: upsample3d_helper(x,size))

class Resnet3DBuilder(object):
    """ResNet3D."""

    iteration = 0 #keeps track of how many models were built in current python session; useful for naming conv4d layer variables
    use_shortcut = True
    use_colearn = True
    
    #returns a list of list of encoder outputs
    #first list is across channels of the input tensor
    #second list is from smallest to largest outputs
    @staticmethod
    def build_encoders(input_tensor, num_channels, block_fn, repetitions, reg_factor, mode):
        
        encoder_outputs = []
        
        #create an encoder for each input channel
        for channel in range(min(num_channels, 2)): 
            encoder_output = []
            
            if num_channels == 4: #for num_channels = 4, we concatenate radiomics channels into PET/CT pairs
                modality_channel = crop(CHANNEL_AXIS, channel, channel+1)(input_tensor)
                radiomics_channel = crop(CHANNEL_AXIS, channel+2, channel+3)(input_tensor)
                input_channel = concatenate([modality_channel, radiomics_channel], CHANNEL_AXIS)
            else:
                input_channel = crop(CHANNEL_AXIS, channel, channel+1)(input_tensor)
                
            block_fn = _get_block(block_fn)

            if mode == 'detection': #downsample for detection
                subsampling_strides = (2,2,2)
                
                # first conv
                input_channel = _conv_bn_relu3D(filters=64, kernel_size=(7, 7, 7),
                                        strides=subsampling_strides,
                                        kernel_regularizer=l2(reg_factor)
                                        )(input_channel)

                input_channel = MaxPooling3D(pool_size=(3, 3, 3), strides=subsampling_strides,
                                 padding="same")(input_channel)

            elif mode == 'segmentation' or mode == 'localisation':
                pass #subsampling_strides = (1, 1, 1)


            # repeat blocks
            block = input_channel
            filters = 64

            for i, r in enumerate(repetitions):
                block = _residual_block3d(block_fn, filters=filters,
                                          kernel_regularizer=l2(reg_factor),
                                          repetitions=r, is_first_layer=(i==0)
                                          )(block)
                filters *= 2
                encoder_output.append(block)
                
            encoder_output.reverse() #outputs are now smallest to largest
            encoder_outputs.append(encoder_output)
                
        return encoder_outputs
    
    @staticmethod
    #accepts a list of tensors n * (b,x,y,z,f,1)
    #returns a (b,x,y,z, n*f) block
    #index refers to the nth encoder output
    def get_colearning_block(encoder_outputs, reg_factor, index):
        def prep_encoder_block(encoder_block):
            block_output = ZeroPadding3D(padding=1)(encoder_block)
            block_output = expand_dims(axis=modality_axis)(block_output)
            return block_output
        
        modality_axis = 1
        num_modalities = len(encoder_outputs)
        
        if num_modalities == 1:
            stacked_features = encoder_outputs[0]
            fusion_map = prep_encoder_block(encoder_outputs[0])
            
        else:
            stacked_features = concatenate(encoder_outputs, CHANNEL_AXIS)
            
            colearning_input = []
            for encoder_block in encoder_outputs:
                colearning_input.append(prep_encoder_block(encoder_block))
            fusion_map = concatenate(colearning_input, modality_axis)
            
        if Resnet3DBuilder.use_colearn:
            fusion_map = Permute((5,1,4,3,2))(fusion_map) #(b,l,w,h,d,c)=>(b,c,l,d,h,w)
            fusion_map = conv4d_wrapper(num_filters=stacked_features._keras_shape[CHANNEL_AXIS], 
                                             kernel_size=(num_modalities,3,3,3),  #experimental conv4d only operates on a channels-first basis
                                              reg_factor=reg_factor,
                                              padding='valid',
                                        index=index)(fusion_map)
            fusion_map = Permute((2,5,4,3,1))(fusion_map) #(b,c,l,d,h,w)=>(b,l,w,h,d,c)
            fusion_map = squeeze(modality_axis)(fusion_map)

            colearning_output = Multiply()([stacked_features, fusion_map])
            
            return colearning_output
        
        else:
            print('colearn off')
            return stacked_features



       
    #encoder outputs are arranged from smallest to largest volume
    @staticmethod
    def get_colearning_blocks(encoder_outputs, reg_factor, mode):
        
        if mode == 'detection'  or mode == 'localisation': #for detection only use the smallest colearning_block
            num_colearn_blocks = 1
        
        elif mode == 'segmentation':
            num_colearn_blocks = len(encoder_outputs[0])

            
        colearn_blocks = []

        for i in range(num_colearn_blocks):
            colearn_block = []
            for encoder_output in encoder_outputs:
                colearn_block.append(_bn_relu(encoder_output[i])) #bn_relu first since encoders just did a conv
            colearn_block = Resnet3DBuilder.get_colearning_block(colearn_block, reg_factor, i)

            colearn_blocks.append(colearn_block)

        return colearn_blocks
    
    
    @staticmethod
    def build_decoder(colearn_blocks, block_fn, repetitions, reg_factor, mode):
        if mode == 'detection' or mode == 'localisation':
            
            #reduce number of filters from 512 * numModalities to 512 so that the dense layer isn't too tired
            conv = _bn_relu_conv3d(filters=512, kernel_size=(1,1,1),
               strides=(1,1,1), padding="valid",
               kernel_initializer="he_normal",
               )(colearn_blocks[0])
            
            return _bn_relu(conv)
        
        elif mode == 'segmentation': #colearn blocks are from smallest to largest
            
            #localisation => increasing numbers of filters
            #segmentation => decreasing numbers of filters
            filters = []
            current_filter_count = 64
            for _ in repetitions:
                filters.append(current_filter_count)
                current_filter_count *= 2
            filters.reverse() 

            block = colearn_blocks[0]
            
            for i, r in enumerate(repetitions):
                if i == 0: #first block is colearn_blocks[0], used directly as input
                    continue

                #residual block up/downsamples for segmentation/localisation respectively
                block = _residual_block3d(block_fn, filters=filters[i], 
                                          kernel_regularizer=l2(reg_factor),
                                          repetitions=r, is_first_layer=(i == 0),
                                          deconv=True)(block)
                block = concatenate([block, colearn_blocks[i]])

            #final activation
            block = _bn_relu(block)

            return block

    @staticmethod
    def build(input_shape, output_shape, block_fn, repetitions, reg_factor, mode, use_shortcut, use_colearn):
                    
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
        Resnet3DBuilder.iteration += 1
        Resnet3DBuilder.use_shortcut = use_shortcut
        Resnet3DBuilder.use_colearn = use_colearn
        print('Model number for session:', Resnet3DBuilder.iteration)
        
        if mode == 'localisation':   
            input = [Input(shape=input_shape[0]), Input(shape=input_shape[1])]
            main_input = input[0]
            num_channels = input_shape[0][CHANNEL_AXIS-1]
            
        elif mode == 'detection' or mode == 'segmentation':
            input = Input(shape=input_shape)
            main_input = input
            num_channels = input_shape[CHANNEL_AXIS-1]

        encoder_outputs = Resnet3DBuilder.build_encoders(main_input, num_channels, block_fn, repetitions, reg_factor, mode)
        colearn_outputs = Resnet3DBuilder.get_colearning_blocks(encoder_outputs, reg_factor, mode)
        
        #print(colearn_outputs)
        
        decoder_output = Resnet3DBuilder.build_decoder(colearn_outputs, block_fn, repetitions, reg_factor, mode)
        
        #print(decoder_output)
        
        if mode == 'detection':
            # average poll and classification
            pool2 = AveragePooling3D(pool_size=(decoder_output._keras_shape[DIM1_AXIS],
                                        decoder_output._keras_shape[DIM2_AXIS],
                                        decoder_output._keras_shape[DIM3_AXIS]),
                             strides=(1, 1, 1))(decoder_output)
            
            flatten1 = Flatten()(pool2) 
            
            dense_centre = Dense(units=output_shape[0],
                          kernel_initializer="he_normal",
                          activation='linear',
                          kernel_regularizer=l2(reg_factor))(flatten1)
            
            model = Model(inputs=input, outputs=dense_centre)
            
        elif mode == 'localisation':

            # average poll and classification
            decoder_output = AveragePooling3D(pool_size=(decoder_output._keras_shape[DIM1_AXIS],
                                        decoder_output._keras_shape[DIM2_AXIS],
                                        decoder_output._keras_shape[DIM3_AXIS]),
                             strides=(1, 1, 1))(decoder_output)
            
            #print(decoder_output)
            
            flatten = Flatten()(decoder_output)

            #centre prediction
            dense_centre = Dense(units=output_shape[0],
                          kernel_initializer="he_normal",
                          activation='linear',
                          kernel_regularizer=l2(reg_factor))(flatten)
            
            dense_centre = Add()([dense_centre, input[1]]) #predicted centre added here
            
            model = Model(inputs=input, outputs=dense_centre)
        
        elif mode == 'segmentation':
            # deconvolution time o_O; 
            # approximately the same layer structure as the convolution half but in reverse
            

            deconv = Conv3D(filters=1, kernel_size=(1,1,1),
                          strides=(1,1,1), kernel_initializer='he_normal',
                          kernel_regularizer=l2(reg_factor))(decoder_output)
            deconv = squeeze(-1)(deconv)

            final_activation = Activation('sigmoid', name='final_activation')(deconv)

            model = Model(inputs=input, outputs=[final_activation])

        else:
            model = None
            print('Wrong mode selected:', mode)

        
        return model

    @staticmethod
    def build_resnet_18(input_shape, output_shape, reg_factor=1e-4, mode='detection', use_shortcut=True, use_colearn=True):
        """Build resnet 18."""
        return Resnet3DBuilder.build(input_shape, output_shape, basic_block, 
                                     [2, 2, 2, 2], reg_factor=reg_factor, mode=mode,
                                    use_shortcut=use_shortcut, use_colearn=use_colearn)

    @staticmethod
    def build_resnet_34(input_shape, output_shape, reg_factor=1e-4, mode='detection', use_shortcut=True, use_colearn=True):
        """Build resnet 34."""
        return Resnet3DBuilder.build(input_shape, output_shape, basic_block,
                                     [3, 4, 6, 3], reg_factor=reg_factor, mode=mode,
                                    use_shortcut=use_shortcut, use_colearn=use_colearn)
