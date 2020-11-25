"""
Modified Unet blocks and constructors
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, \
    Activation, BatchNormalization, Add


def conv_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)    
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
    

def down_layer(input_tensor, n_filters, kernel_size=3, batchnorm=True,dropout=0):
    conv = conv_block(input_tensor, n_filters, kernel_size, batchnorm)
    pool = MaxPooling2D((2, 2))(conv)
    if dropout > 0 and dropout < 1:
        drop = Dropout(dropout)(pool)
    else:
        drop = pool
    
    return conv, drop


def up_layer(input_tensor, concat_tensor, n_filters, kernel_size=3, batchnorm=True,dropout=0):
    decv = Conv2DTranspose(n_filters, (kernel_size, kernel_size), strides=(2, 2), padding='same')(input_tensor)
    conc = concatenate([decv, concat_tensor])
    if dropout > 0 and dropout < 1:
        drop = Dropout(dropout)(conc)
    else:
        drop = conc
    conv = conv_block(drop, n_filters, kernel_size, batchnorm)
    
    return conv


def unet(pretrained_weights=None, input_size=(256, 256, 3), n_output=1, n_base=16,batchnorm=True,dropout=0,n_layers=4):
    # input layer
    in1 = Input(input_size)
    
    concat_list = []
    inp = in1
    
    # downward length, saving conv layers for concat
    for i in range(n_layers):
        n_filt = n_base * 2**i
        c, d = down_layer(inp,n_filters=n_filt,batchnorm=batchnorm,dropout=dropout)
        inp = d
        concat_list.append(c)
    # convolution at bottom
    enc_out = conv_block(inp, n_base*16,batchnorm=batchnorm)

    # upward length
    inp = enc_out
    for i in range(n_layers):
        n_filt = n_base * 2**(n_layers-i-1)
        u = up_layer(inp,concat_list.pop(),n_filters=n_filt,batchnorm=batchnorm,dropout=dropout)
        inp = u
    
    # output layer
    ou1 = Conv2D(n_output, (1, 1), activation='softmax')(inp)
    
    model = Model([in1], [ou1])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
