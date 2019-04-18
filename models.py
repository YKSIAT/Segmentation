from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K


def bn_re(input_):
    norm = BatchNormalization()(input_)
    return Activation("relu")(norm)


def squeeze_excite_block(input_, ratio=16):
    """
        Create a channel-wise squeeze-excite block
    Args:
        input_: input tensor
        ratio:
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    # print("filters", filters)
    se_shape = (1, 1, filters)

    se1 = GlobalAveragePooling2D()(init)
    se2 = Reshape(se_shape)(se1)
    se3 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se2)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se3)
    #
    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def squeeze_excite_block_plot(input_size=(256, 256, 4)):
    inputs = Input(input_size, name="inputs")
    input_bn = bn_re(inputs)
    input_se = squeeze_excite_block(input_bn)
    con11 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name="")(input_se)
    con11_bn = bn_re(con11)
    con11_se = squeeze_excite_block(con11_bn)
    con12 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name="")(con11_se)
    con12_bn = bn_re(con12)
    con12_se = squeeze_excite_block(con12_bn)
    con3 = Conv2D(1, 1, activation='sigmoid', name="")(con12_se)
    model = Model(input=inputs, output=con3)

    return model


def spatial_squeeze_excite_block(input_tensor):
    ''' Create a spatial squeeze-excite block
    Args:
        input_tensor: input tensor
    Returns: a keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x


def channel_spatial_squeeze_excite(input_tensor, ratio=16):
    ''' Create a spatial squeeze-excite block
    Args:
        input_tensor: input tensor
        ratio:
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)

    x = add([cse, sse])
    return x


def unet_scSE_gap(input_size=(240, 240, 4)):
    inputs = Input(input_size)
    input_bn = bn_re(inputs)
    con11 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(input_bn)
    con11_bn = bn_re(con11)
    con11_se = channel_spatial_squeeze_excite(con11_bn)
    con12 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(con11_se)
    con12_bn = bn_re(con12)
    con12_se = channel_spatial_squeeze_excite(con12_bn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(con12_se)
    con21 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    con21_bn = bn_re(con21)
    con21_se = channel_spatial_squeeze_excite(con21_bn)
    con22 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(con21_se)
    con22_bn = bn_re(con22)
    con22_se = channel_spatial_squeeze_excite(con22_bn)
    pool2 = MaxPooling2D(pool_size=(2, 2))(con22_se)
    con31 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    con31_bn = bn_re(con31)
    con31_se = channel_spatial_squeeze_excite(con31_bn)
    con32 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(con31_se)
    con32_bn = bn_re(con32)
    con32_se = channel_spatial_squeeze_excite(con32_bn)
    pool3 = MaxPooling2D(pool_size=(2, 2))(con32_se)
    con41 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    con41_bn = bn_re(con41)
    con41_se = channel_spatial_squeeze_excite(con41_bn)
    con42 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(con41_se)
    con42_bn = bn_re(con42)
    con42_se = channel_spatial_squeeze_excite(con42_bn)
    drop4 = Dropout(0.5)(con42_se)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    con51 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    con51_bn = bn_re(con51)
    gap5 = GlobalAveragePooling2D()(con51_bn)
    con52 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(con51_bn)
    con52_bn = bn_re(con52)
    # drop5 = Dropout(0.5)(con52_bn)
    gap2 = multiply([con52_bn, gap5])
    up6 = Conv2D(512, 2, padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(gap2))
    up6_bn = bn_re(up6)
    merge6 = concatenate([drop4, up6_bn], axis=3)
    con61 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
    con61_bn = bn_re(con61)
    con61_se = channel_spatial_squeeze_excite(con61_bn)
    con62 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(con61_se)
    con62_bn = bn_re(con62)
    con62_se = channel_spatial_squeeze_excite(con62_bn)
    up7 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(con62_se))
    up7_bn = bn_re(up7)
    merge7 = concatenate([con32_bn, up7_bn], axis=3)
    con71 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
    con7_bn = bn_re(con71)
    con71_se = channel_spatial_squeeze_excite(con7_bn)
    con72 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(con71_se)
    con72_bn = bn_re(con72)
    con72_se = channel_spatial_squeeze_excite(con72_bn)

    up8 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(con72_se))
    up8_bn = bn_re(up8)
    merge8 = concatenate([con22_bn, up8_bn], axis=3)
    con81 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
    con81_bn = bn_re(con81)
    con81_se = channel_spatial_squeeze_excite(con81_bn)
    con82 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(con81_se)
    con82_bn = bn_re(con82)
    con82_se = channel_spatial_squeeze_excite(con82_bn)
    up9 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(con82_se))
    up9_bn = bn_re(up9)
    merge9 = concatenate([con12_bn, up9_bn], axis=3)
    con91 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
    con91_bn = bn_re(con91)
    con92 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(con91_bn)
    con92_bn = bn_re(con92)
    con93 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(con92_bn)
    con93_bn = bn_re(con93)
    con10 = Conv2D(1, 1, activation='sigmoid')(con93_bn)
    model = Model(input=inputs, output=con10)
    # model.load_weights("brats_se_epoch_300.h5")
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model


def bn_sig(input_):
    norm = BatchNormalization()(input_)
    return Activation("sigmoid")(norm)


def double_con_bn_re(filters, layer):
    con1 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(layer)
    bn1 = bn_re(con1)
    con2 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(bn1)
    bn = bn_re(con2)
    return bn


def channel_attention(input_layers, ratio=16):
    """

    :param input_layers:
    :param ratio:
    :return:
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input_layers._keras_shape[channel_axis]
    ca_shape = (1, 1, filters)
    ca1 = GlobalAveragePooling2D()(input_layers)
    ca2 = Reshape(ca_shape)(ca1)
    ca3 = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(ca2)
    ca4 = bn_re(ca3)
    ca5 = Dense(filters, kernel_initializer='he_normal', use_bias=False)(ca4)
    ca = bn_sig(ca5)
    if K.image_data_format() == 'channels_first':
        ca = Permute((3, 1, 2))(ca)

    x = multiply([input_layers, ca])
    return x


def multi_level_attention(input_layers):
    """

    :param input_layers:
    :return:
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input_layers._keras_shape[channel_axis]
    pool_layer_1 = MaxPooling2D(pool_size=(2, 2))(input_layers)    # 32
    d_con_1 = double_con_bn_re(filters, pool_layer_1)    # 64
    pool_layer_2 = MaxPooling2D(pool_size=(2, 2))(d_con_1)    # 32
    d_con_layer_2 = double_con_bn_re(filters, pool_layer_2)    # 64
    up_con_layer_1 = Conv2D(filters, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))
                                                                                        (d_con_layer_2))
    add_node_1 = add([d_con_1, up_con_layer_1])    # 64
    con_1 = Conv2D(filters, 2, padding='same', kernel_initializer='he_normal')(add_node_1)
    up_con_layer_2 = Conv2D(filters, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))
                                                                                        (con_1))
    add_node_2 = add([input_layers, up_con_layer_2])
    d_con_layer_3 = double_con_bn_re(filters, add_node_2)
    out_layer = multiply([d_con_layer_3, input_layers])
    return out_layer


def channel_space_block(input_layer):
    node_1 = channel_attention(input_layer, ratio=16)
    node_2 = multi_level_attention(input_layer)
    output_layer = add([node_1, node_2])
    return output_layer


def plot_test_net(input_size=(240, 240, 4)):
    inputs = Input(input_size)
    con11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="con11")(inputs)
    con12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="con12")(con11)
    ca_layer = channel_space_block(con12)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(ca_layer)
    con21 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="con21")(pool1)
    con22 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="con22")(con21)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(con22)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool4")(pool2)
    model = Model(input=inputs, output=pool4)
    return model


def feature_attention_net(input_size=(240, 240, 4)):
    inputs = Input(input_size)
    input_bn = bn_re(inputs)
    con11 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(input_bn)
    con11_bn = bn_re(con11)
    con11_se = channel_space_block(con11_bn)
    con12 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(con11_se)
    con12_bn = bn_re(con12)
    con12_se = channel_space_block(con12_bn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(con12_se)
    con21 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    con21_bn = bn_re(con21)
    con21_se = channel_space_block(con21_bn)
    con22 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(con21_se)
    con22_bn = bn_re(con22)
    con22_se = channel_space_block(con22_bn)
    pool2 = MaxPooling2D(pool_size=(2, 2))(con22_se)
    con31 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    con31_bn = bn_re(con31)
    con31_se = channel_space_block(con31_bn)
    con32 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(con31_se)
    con32_bn = bn_re(con32)
    con32_se = channel_space_block(con32_bn)
    pool3 = MaxPooling2D(pool_size=(2, 2))(con32_se)
    con41 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    con41_bn = bn_re(con41)
    # con41_se = channel_space_block(con41_bn)
    con42 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(con41_bn)
    con42_bn = bn_re(con42)
    # con42_se = channel_space_block(con42_bn)
    drop4 = Dropout(0.5)(con42_bn)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    con51 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    con51_bn = bn_re(con51)
    gap5 = GlobalAveragePooling2D()(con51_bn)
    con52 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(con51_bn)
    con52_bn = bn_re(con52)
    # drop5 = Dropout(0.5)(con52_bn)
    gap2 = multiply([con52_bn, gap5])
    up6 = Conv2D(512, 2, padding='same', kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(gap2))
    up6_bn = bn_re(up6)
    merge6 = concatenate([drop4, up6_bn], axis=3)
    con61 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
    con61_bn = bn_re(con61)
    # con61_se = channel_space_block(con61_bn)
    con62 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(con61_bn)
    con62_bn = bn_re(con62)
    # con62_se = channel_space_block(con62_bn)
    up7 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(con62_bn))
    up7_bn = bn_re(up7)
    merge7 = concatenate([con32_bn, up7_bn], axis=3)
    con71 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
    con7_bn = bn_re(con71)
    con71_se = channel_space_block(con7_bn)
    con72 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(con71_se)
    con72_bn = bn_re(con72)
    con72_se = channel_space_block(con72_bn)

    up8 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(con72_se))
    up8_bn = bn_re(up8)
    merge8 = concatenate([con22_bn, up8_bn], axis=3)
    con81 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
    con81_bn = bn_re(con81)
    con81_se = channel_space_block(con81_bn)
    con82 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(con81_se)
    con82_bn = bn_re(con82)
    con82_se = channel_space_block(con82_bn)
    up9 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(con82_se))
    up9_bn = bn_re(up9)
    merge9 = concatenate([con12_bn, up9_bn], axis=3)
    con91 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
    con91_bn = bn_re(con91)
    con92 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(con91_bn)
    con92_bn = bn_re(con92)
    con93 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(con92_bn)
    con93_bn = bn_re(con93)
    con10 = Conv2D(1, 1, activation='sigmoid')(con93_bn)
    model = Model(input=inputs, output=con10)

    return model

    pass


def plot_model_(p_model):
    # model_ = unet()

    # model_ = squeeze_excite_block_plot()
    plot_model(p_model, to_file='./structure/unet_scSE_gap.pdf', show_shapes=True)


if __name__ == "__main__":
    pass
    # model = unet()
    # model.summary()
    plot_model_(unet_scSE_gap())









