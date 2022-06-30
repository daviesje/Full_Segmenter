'''
architecture
'''
import tensorflow as tf
from tensorflow.keras.layers import *

def conv_block(input_tensor, n_filters, kernel_size, strides=(1,1), activation='relu', padding='same', kernel_initializer="he_normal"):
    x = Conv2D(n_filters, kernel_size, strides=strides, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(input_tensor)
    x = BatchNormalization()(x)

    return x


def layer_down(input_tensor, n_filters, kernel_size=(3,3)):

    conv = conv_block(input_tensor, n_filters, kernel_size)
    conv = Dropout(0.2)(conv)
    conv = conv_block(conv, n_filters, kernel_size)
    pool = MaxPooling2D((2, 2))(conv)

    return conv, pool

def layer_up(input_tensor, concat_tensor, n_filter, kernel_size=(3,3)):
    decv = concatenate([Conv2DTranspose(n_filter, kernel_size, strides=(2,2), padding='same')(input_tensor), concat_tensor])
    decv = conv_block(decv, n_filter, kernel_size)
    decv = Dropout(0.2)(decv)
    decv = conv_block(decv, n_filter, kernel_size)

    return decv

def MultiResBlock(n_filters, input_tensor, alpha = 1.67, dropout=None):
    
    W = alpha * n_filters
    shortcut = input_tensor

    shortcut = conv_block(shortcut, int(W*0.167) + int(W*0.333) +
                          int(W*0.5), (1, 1), activation=None, padding='same')

    conv3x3 = conv_block(input_tensor, int(W*0.167), (3, 3),
                         activation='relu', padding='same')

    conv5x5 = conv_block(conv3x3, int(W*0.333), (3, 3),
                         activation='relu', padding='same')

    conv7x7 = conv_block(conv5x5, int(W*0.5), (3, 3),
                         activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    if dropout: out = Dropout(dropout)(out)
    out = BatchNormalization(axis=3)(out)

    return out

def DCBlock(n_filters, input_tensor, alpha = 1.67, dropout=None):

    W = alpha * n_filters

    conv3x3 = conv_block(input_tensor, int(W*0.167), (3, 3),
                         activation='relu', padding='same')

    conv5x5 = conv_block(conv3x3, int(W*0.333), (3, 3),
                         activation='relu', padding='same')

    conv7x7 = conv_block(conv5x5, int(W*0.5), (3, 3),
                         activation='relu', padding='same')

    out_1 = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out_1 = BatchNormalization(axis=3)(out_1)

    conv3x3_2 = conv_block(input_tensor, int(W*0.167), (3, 3),
                         activation='relu', padding='same')
    
    conv5x5_2 = conv_block(conv3x3_2, int(W*0.333), (3, 3),
                         activation='relu', padding='same')

    conv7x7_2 = conv_block(conv5x5_2, int(W*0.5), (3, 3),
                         activation='relu', padding='same')

    out_2 = concatenate([conv3x3_2, conv5x5_2, conv7x7_2], axis=3)
    out_2 = BatchNormalization(axis=3)(out_2)

    out = add([out_1, out_2])
    out = Activation('relu')(out)
    if dropout: out = Dropout(dropout)(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(n_filters, length, input_tensor):
    shortcut = input_tensor
    shortcut = conv_block(shortcut, n_filters, (1, 1),
                         activation=None, padding='same')

    out = conv_block(input_tensor, n_filters, (3, 3), activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv_block(shortcut, n_filters, (1, 1),
                              activation=None, padding='same')

        out = conv_block(out, n_filters, (3, 3), activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out

def Reslayer_down(input_tensor, n_filters, length, dropout=None):
    conv = MultiResBlock(n_filters, input_tensor, dropout=None)
    pool = MaxPooling2D((2,2))(conv)
    if dropout: pool = Dropout(dropout)(pool)
    conv = ResPath(n_filters, length, input_tensor)

    return conv, pool

def Reslayer_up(input_tensor, concat_tensor, n_filters, dropout=None):
    decv = concatenate([Conv2DTranspose(n_filters, (2,2), strides=(2,2), padding='same')(input_tensor), concat_tensor])
    decv = MultiResBlock(n_filters, decv, dropout=dropout)

    return decv

def DClayer_down(input_tensor, n_filters, length, dropout=None):
    conv = DCBlock(n_filters, input_tensor, dropout=None)
    pool = MaxPooling2D((2,2))(conv)
    if dropout: pool = Dropout(dropout)(pool)
    conv = ResPath(n_filters, length, input_tensor)

    return conv, pool

def DClayer_up(input_tensor, concat_tensor, n_filters, dropout=None):
    decv = concatenate([Conv2DTranspose(n_filters, (2,2), strides=(2,2), padding='same')(input_tensor), concat_tensor])
    decv = DCBlock(n_filters, decv, dropout=dropout)

    return decv

def u_net(input_size=(128,128,3), n_output=1, n_base=16, pretrained_weights=None):
    #Input Layer
    img_input = Input(input_size)

    c1, d1 = layer_down(img_input, n_base)
    c2, d2 = layer_down(d1, n_base*2)
    c3, d3 = layer_down(d2, n_base*4)
    c4, d4 = layer_down(d3, n_base*8)

    c5 = conv_block(d4, n_base*16, kernel_size=(3,3))

    u4 = layer_up(c5, c4, n_base*8)
    u3 = layer_up(u4, c3, n_base*4)
    u2 = layer_up(u3, c2, n_base*2)
    u1 = layer_up(u2, c1, n_base)

    out = Conv2D(n_output, (1, 1) , padding='same',activation='sigmoid')(u1)

    model = tf.keras.Model(inputs=[img_input], outputs=[out])
    #model.summary()

    if(pretrained_weights):
       model.load_weights(pretrained_weights)

    return model

def mRES_net(input_size=(128,128,3), n_output=1, n_base=24, dropout=None, pretrained_weights=None):
    """
    MultiRes UNet: https://arxiv.org/abs/1902.04049
    """
    img_input = Input(input_size)

    c1, d1 = Reslayer_down(img_input, n_base, 5)
    c2, d2 = Reslayer_down(d1, n_base*2,  4, dropout=dropout)
    c3, d3 = Reslayer_down(d2, n_base*4,  3)
    c4, d4 = Reslayer_down(d3, n_base*8,  2, dropout=dropout)
    c5, d5 = Reslayer_down(d4, n_base*16, 1)

    c6 = MultiResBlock(n_base*32, d5)

    u5 = Reslayer_up(c6, c5, n_base*16)
    u4 = Reslayer_up(u5, c4, n_base*8, dropout=dropout)
    u3 = Reslayer_up(u4, c3, n_base*4)
    u2 = Reslayer_up(u3, c2, n_base*2, dropout=dropout)
    u1 = Reslayer_up(u2, c1, n_base)

    out = Conv2D(n_output, (1, 1) , padding='same',activation='softmax')(u1)

    model = tf.keras.Model(inputs=[img_input], outputs=[out])

    if(pretrained_weights):
       model.load_weights(pretrained_weights)

    return model

def DCu_net(input_size=(128,128,3), n_output=1, n_base=24, dropout=None, pretrained_weights=None):
    """
    DCUNet: https://arxiv.org/abs/2006.00414
    """
    img_input = Input(input_size)

    c1, d1 = DClayer_down(img_input, n_base, 5)
    c2, d2 = DClayer_down(d1, n_base*2,  4, dropout=dropout)
    c3, d3 = DClayer_down(d2, n_base*4,  3)
    c4, d4 = DClayer_down(d3, n_base*8,  2, dropout=dropout)
    c5, d5 = DClayer_down(d4, n_base*16, 1)

    c6 = DCBlock(n_base*32, d5)

    u5 = DClayer_up(c6, c5, n_base*16)
    u4 = DClayer_up(u5, c4, n_base*8, dropout=dropout)
    u3 = DClayer_up(u4, c3, n_base*4)
    u2 = DClayer_up(u3, c2, n_base*2, dropout=dropout)
    u1 = DClayer_up(u2, c1, n_base)

    out = Conv2D(n_output, (1, 1) , padding='same',activation='softmax')(u1)

    model = tf.keras.Model(inputs=[img_input], outputs=[out])

    if(pretrained_weights):
       model.load_weights(pretrained_weights)

    return model


