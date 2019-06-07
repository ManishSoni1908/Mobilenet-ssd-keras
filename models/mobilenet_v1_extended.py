import sys
sys.path.append("/home/manish/MobileNet-ssd-keras/models")
import keras
import numpy as np 
import cv2
import keras.backend as K
import keras.layers as KL
from depthwise_conv2d import DepthwiseConvolution2D
from keras.models import Model
from keras.layers import Input, Lambda, Activation,Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,BatchNormalization, Add, Conv2DTranspose
from keras.regularizers import l2

def mobilenet(input_tensor):

    if input_tensor is None:
        input_tensor = Input(shape=(300,300,3))


    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_padding')(input_tensor)
    x = Convolution2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv0")(x)

    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name = "conv0/bn")(x)

    x = Activation('relu')(x)



    x = DepthwiseConvolution2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False, name="conv1/dw")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv1/dw/bn")(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False, name="conv1")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv1/bn")(x)
    x = Activation('relu')(x)

    print ("conv1 shape: ", x.shape)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv2_padding')(x)
    x = DepthwiseConvolution2D(64, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv2/dw")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv2/dw/bn")(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv2")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv2/bn")(x)
    x = Activation('relu')(x)

    

    x = DepthwiseConvolution2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,name="conv3/dw")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv3/dw/bn")(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv3")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv3/bn")(x)
    x = Activation('relu')(x)

    print ("conv3 shape: ", x.shape)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv3_padding')(x)
    x = DepthwiseConvolution2D(128, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv4/dw")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv4/dw/bn")(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv4")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv4/bn")(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False,name="conv5/dw")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv5/dw/bn")(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv5")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv5/bn")(x)
    x = Activation('relu')(x)


    conv5 = x

    print ("conv5 shape: ", x.shape)

    
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv4_padding')(x)
    x = DepthwiseConvolution2D(256, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv6/dw")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv6/dw/bn")(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv6")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv6/bn")(x)
    x = Activation('relu')(x)

  
    
    for i in range(5):
        x = DepthwiseConvolution2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False,name=("conv" + str(7+i)+"/dw" ))(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name=("conv" + str(7+i)+"/dw/bn" ))(x)
        x = Activation('relu')(x)
        x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False,name=("conv" + str(7+i)))(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name=("conv" + str(7+i) +"/bn"))(x)
        x = Activation('relu')(x)

    # print ("conv11 shape: ", x.shape)
    conv11 = x


    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv5_padding')(x)
    x = DepthwiseConvolution2D(512, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv12/dw")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv12/dw/bn")(x)
    x = Activation('relu')(x)
    x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv12")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv12/bn")(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False,name="conv13/dw")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv13/dw/bn")(x)
    x = Activation('relu')(x)
    x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv13")(x)
    x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv13/bn")(x)
    x = Activation('relu')(x)

    conv13 = x

    # print ("conv13 shape: ", x.shape)


    # model = Model(inputs=input_tensor, outputs=x)

    return [conv5,conv11,conv13]
