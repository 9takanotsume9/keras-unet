import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from keras.layers import MaxPooling2D, UpSampling2D, Cropping2D, Reshape
from keras.optimizers import Adam


def encoding(filters, kernel_size, net):
    net = ZeroPadding2D((1, 1))(net)
    net = LeakyReLU(0.2)(net)
    net = Conv2D(filters, kernel_size, strides=2, padding='valid')(net)
    net = BatchNormalization()(net)
    return net


def decoding(filters, add_dropout, net):
    net = Activation(activation='relu')(net)
    net = Conv2DTranspose(filters, 2, strides=2, kernel_initializer='he_uniform')(net)
    net = BatchNormalization()(net)
    if add_dropout:
        net = Dropout(0.5)(net)
    return net


def cropping(target, refer):
    # width, the 3rd dimension
    cw = (target._keras_shape[2] - refer._keras_shape[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target._keras_shape[1] - refer._keras_shape[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    return 2.0*intersection / (K.sum(y_true)+K.sum(y_pred)+1)


def unet_loss(y_true, y_pred):
    cross_entropy = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    
    return cross_entropy


def UNET(num_classes=1):
    inputs = Input((360, 640, 3))
    enc1 = ZeroPadding2D((1, 1))(inputs)
    enc1 = Conv2D(32, 4, strides=2)(enc1)
    enc2 = encoding(32*2, 4, enc1)
    enc3 = encoding(32*4, 4, enc2)
    enc4 = encoding(32*8, 4, enc3)
    enc5 = encoding(32*8, 4, enc4)
    enc6 = encoding(32*8, 4, enc5)
    enc7 = encoding(32*8, 4, enc6)
    
    dec7 = decoding(32*8, True, enc7)
    ch, cw = cropping(enc6, dec7)
    crop_enc6 = Cropping2D(cropping=(ch, cw))(enc6)
    dec7 = concatenate([dec7, crop_enc6], axis=-1)
    dec6 = decoding(32*8, True, dec7)
    dec6 = Conv2DTranspose(32*8, (4, 1), strides=1, kernel_initializer='he_uniform')(dec6)
    dec6 = concatenate([dec6, enc5], axis=-1)
    dec5 = decoding(32*8, True, dec6)
    dec5 = concatenate([dec5, enc4], axis=-1)
    dec4 = decoding(32*4, True, dec5)
    dec4 = Conv2DTranspose(32*4, (2, 1), strides=1, kernel_initializer='he_uniform')(dec4)
    dec4 = concatenate([dec4, enc3], axis=-1)
    dec3 = decoding(32*2, True, dec4)
    dec3 = concatenate([dec3, enc2], axis=-1)
    dec2 = decoding(32, True, dec3)
    dec2 = concatenate([dec2, enc1], axis=-1)
    dec1 = Activation('relu')(dec2)
    dec1 = Conv2DTranspose(num_classes, 2, strides=2, kernel_initializer='he_uniform')(dec1)
    outputs = Activation('sigmoid')(dec1)

    return Model(inputs=inputs, outputs=outputs)
