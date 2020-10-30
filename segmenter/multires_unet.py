"""
Modified multires Unet blocks and constructors
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, \
    Activation, BatchNormalization, Add


def multires_block(input_tensor,kernel_size=3,n_filters=24,batchnorm=True):
    l1 = Conv2D(filters=n_filters//6,kernel_size=(kernel_size,kernel_size), kernel_initializer="he_normal",
                padding="same",activation='relu')(input_tensor)
                
    if batchnorm:
        l1 = BatchNormalization()(l1)
    
    l2 = Conv2D(filters=n_filters//3,kernel_size=(kernel_size,kernel_size), kernel_initializer="he_normal",
                padding="same",activation='relu')(2)
                
    if batchnorm:
        l2 = BatchNormalization()(l2)
    
    l3 = Conv2D(filters=n_filters//2,kernel_size=(kernel_size,kernel_size), kernel_initializer="he_normal",
                padding="same",activation='relu')(l2)
                
    if batchnorm:
        l3 = BatchNormalization()(l3)

    conc = concatenate([l1,l2,l3])

    res = Conv2D(filters=n_filters,kernel_size=(1,1), kernel_initializer="he_normal",
                padding="same",activation='relu')(input_tensor)

    if batchnorm:
        res = BatchNormalization()(res)

    out = Add()([conc,res])

    return out


def res_path(input_tensor,n_filters=24,n_blocks=4,batchnorm=True):
    inl = input_tensor
    for i in range(n_blocks):
        c = Conv2D(filters=n_filters,kernel_size=(3,3), kernel_initializer="he_normal",
                    padding="same",activation='relu')(inl)
                    
        if batchnorm:
            c = BatchNormalization()(c)

        r = Conv2D(filters=n_filters,kernel_size=(1,1), kernel_initializer="he_normal",
                    padding="same",activation='relu')(inl)

        if batchnorm:
            r = BatchNormalization()(r)

        a = Add()([c,r])
        inl = a

    return a


def down_layer(input_tensor, kernel_size=3, n_filters=24, batchnorm=True,dropout=0):
    conv = multires_block(input_tensor,kernel_size=kernel_size,n_filters=n_filters,batchnorm=batchnorm)
    pool = MaxPooling2D((2, 2))(conv)

    if dropout > 0 and dropout < 1:
        drop = Dropout(dropout)(pool)
    else:
        drop = pool
    
    return conv, drop


def up_layer(input_tensor, concat_tensor, n_filters=24 , n_blocks=4, kernel_size=3,batchnorm=True,dropout=0):
    decv = Conv2DTranspose(1, (kernel_size, kernel_size), strides=(2, 2), padding='same',activation='relu')(input_tensor)

    if batchnorm:
        decv = BatchNormalization()(decv)
    
    path = res_path(concat_tensor,n_filters=n_filters,n_blocks=n_blocks,batchnorm=batchnorm)
    conc = concatenate([decv,path])
    if dropout > 0 and dropout < 1:
        drop = Dropout(dropout)(conc)
    else:
        drop = conc

    conv = multires_block(drop, n_filters=n_filters, kernel_size=kernel_size,batchnorm=batchnorm)
    
    return conv
    
def mr_unet(pretrained_weights=None, input_size=(256, 256, 3), n_output=1, n_base=24,batchnorm=True,dropout=0,n_layers=4):
    # input layer
    in1 = Input(input_size)

    in2 = Conv2D(1,(3,3),activation='relu',padding='same')(in1)
    
    # downward length, saving conv layers for concat
    concat_list = []
    inp = in2
    for i in range(n_layers):
        n_filt = n_base * 2**i
        c, d = down_layer(inp,n_filters=n_filt,batchnorm=batchnorm,dropout=dropout)
        inp = d
        concat_list.append(c)
    
    # convolution at bottom
    enc_out = multires_block(d6,n_filters=n_base*16,batchnorm=batchnorm)

    # upward length
    inp = enc_out
    for i in range(n_layers):
        n_filt = n_base * 2**(n_layers-i-1)
        u = up_layer(inp,concat_list.pop(),n_blocks=n_layers-i,n_filters=n_filt,batchnorm=batchnorm,dropout=dropout)
        inp = u
        
    # output layer
    ou1 = Conv2D(n_output, (1, 1), activation='sigmoid',padding='same')(inp)
    
    model = Model([in1], [ou1])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model