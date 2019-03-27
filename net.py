import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *

# Based off the Fully Convolutional Network architecture:
# https://arxiv.org/abs/1605.06211
# https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.htmlx

def FCN8(input_height=512,input_width=512, image_depth=3):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height%32 == 0
    assert input_width%32 == 0

    img_input = Input(shape=(input_height,input_width, image_depth)) # 512x512ximage_depth

    ## Layer 1 input=(512x512ximage_depth), output=(256x256x64)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format="channels_last")(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format="channels_last")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format="channels_last")(x)
    f1 = x

    # Layer 2 input=(256x256x64), output=(128x128x128)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format="channels_last")(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format="channels_last")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format="channels_last")(x)
    f2 = x

    # Layer 3 input=(128x128x128), output=(64x64x256)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format="channels_last")(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format="channels_last")(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format="channels_last")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format="channels_last")(x)
    pool3 = x

    # Layer 4 input=(64x64x256), output=(32x32x512)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format="channels_last")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format="channels_last")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format="channels_last")(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format="channels_last")(x)## (None, 14, 14, 512)

    # Layer 5 input=(32x32x512), output=(16x16x512)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format="channels_last")(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format="channels_last")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format="channels_last")(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format="channels_last")(x)## (None, 7, 7, 512)

    # input=(16x16x512), output=(16x16x4096)
    o = (Conv2D(4096,(7 ,7), activation='relu', padding='same', name="conv6", data_format="channels_last"))(pool5)
    conv7 = (Conv2D(4096,(1 ,1), activation='relu', padding='same', name="conv7", data_format="channels_last"))(o)

    # single class detection (hematoma)
    nClasses = 1

    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(nClasses, kernel_size=(4,4), strides=(4,4), use_bias=False, data_format="channels_last")(conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = (Conv2D(nClasses, (1,1), activation='relu', padding='same', name="pool4_11", data_format="channels_last"))(pool4)
    pool411_2 = (Conv2DTranspose(nClasses , kernel_size=(2,2), strides=(2,2), use_bias=False, data_format="channels_last"))(pool411)

    pool311 = (Conv2D(nClasses, (1,1), activation='relu', padding='same', name="pool3_11", data_format="channels_last"))(pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(nClasses, kernel_size=(8,8),  strides=(8,8), use_bias=False, data_format="channels_last")(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)

    return model
