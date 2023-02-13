import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.layers import LeakyReLU, ReLU, Concatenate

def downsample(filters, size, strides, apply_batchnorm = True):
    '''
    Creates a downsampler, returning a Sequential Keras model.
    
    filters: int, number of convolutional filters to use in Conv2D transformation.
    size: int or 2-tuple, size of convolutional kernel/window to use in Conv2D transformation.
    strides: int or 2-tuple, size of strides to use in Conv2D transformation.
    apply_batchnorm: bool, whether to apply BatchNormalization in downsampler. Default True.
    '''
    
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Make sequential model.
    result = Sequential()
    
    # Add Conv2D.
    result.add(Conv2D(filters, size, strides = strides, padding = 'same',
                             kernel_initializer = initializer, use_bias = False))

    # Optionally add batchnorm.
    if apply_batchnorm:
        result.add(BatchNormalization())

    # Add leaky relu.
    result.add(LeakyReLU(alpha = 0.2))

    return result


def upsample(filters, size, strides, apply_dropout = False):
    '''
    Creates an upsampler, returning a Sequential Keras model.
    
    filters: int, number of convolutional filters to use in Conv2DTranspose transformation.
    size: int or 2-tuple, size of convolutional kernel/window to use in Conv2DTranspose transformation.
    strides: int or 2-tuple, size of strides to use in Conv2DTranspose transformation.
    apply_dropout: bool, whether to apply Dropout in upsampler. Default False.
    '''
    
    initializer = tf.random_normal_initializer(0., 0.02)

    # Make sequential model.
    result = Sequential()
    
    # Add deconv layer (conv2dtranspose).
    result.add(Conv2DTranspose(filters, size, strides = strides,
                                    padding='same',
                                    kernel_initializer = initializer,
                                    use_bias = False))

    # Add batchnorm.
    result.add(BatchNormalization())

    # Optionally add dropout.
    if apply_dropout:
        result.add(Dropout(0.5))
    
    # Add relu.
    result.add(ReLU())

    return result

def create_generator():
    '''
    Creates U-net autoencoder for the generator. Returns a Keras Model.
    '''
    
    # Define input shape.
    inputs = Input(shape = [256, 256, 1])
    
    # Define downsampling layers.
    down_stack = [
        downsample(64, 4, 2, apply_batchnorm = False),
        downsample(128, 4, 2),
        downsample(256, 4, 2),
        downsample(512, 4, 2),
        downsample(512, 4, 2),
        downsample(512, 4, 2),
        downsample(512, 4, 2),
        downsample(512, 4, 2)
    ]
    
    # Define upsampling layers.
    up_stack = [
        upsample(512, 4, 2, apply_dropout = True),
        upsample(512, 4, 2, apply_dropout = True),
        upsample(512, 4, 2, apply_dropout = True),
        upsample(512, 4, 2),
        upsample(256, 4, 2),
        upsample(128, 4, 2),
        upsample(64, 4, 2),
    ]
    
    # Last layer.
    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(1, 4, strides = 2, padding = 'same', kernel_initializer = initializer, activation = 'tanh')
    
    x = inputs

    # Downsampling through the model.
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections.
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = last(x)

    return Model(inputs = inputs, outputs = x)

def create_discriminator():
    '''
    Creates a discriminator.
    '''
    
    initializer = tf.random_normal_initializer(0., 0.02)

    # Define inputs.
    inp = Input(shape = [256, 256, 1], name = 'sketch')
    tar = Input(shape = [256, 256, 1], name = 'target')

    # Concatenate inputs.
    x = Concatenate()([inp, tar])

    # Downsampling.
    down1 = downsample(64, 4, 2, False)(x)
    down2 = downsample(128, 4, 2)(down1)
    down3 = downsample(256, 4, 2)(down2)
    down4 = downsample(512, 4, 1)(down3)

    # Apply Conv2D on last layer.
    last = Conv2D(1, 4, strides = 1, kernel_initializer = initializer)(down4)

    return Model(inputs = [inp, tar], outputs = last)