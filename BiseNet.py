from keras.layers import *
from keras.applications.xception import Xception,preprocess_input
from keras.models import Model
from keras.layers import Conv2D, Input, multiply, Concatenate, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, UpSampling2D
from keras.utils import plot_model
from config import *



def bn_act(inputs, activation='relu'):
    """

    :param inputs:
    :param activation:
    :return:
    """
    bn_layers = BatchNormalization()(inputs)
    return Activation(activation)(bn_layers)


def con_bn_act(inputs, n_filters=64, kernel=(2, 2), strides=1, activation='relu'):
    """

    :param inputs:
    :param n_filters:
    :param kernel:
    :param strides:
    :param activation:
    :return:
    """
    con_layer = Conv2D(n_filters, kernel_size=kernel, kernel_initializer='he_normal', strides=strides)(inputs)
    return bn_act(con_layer, activation=activation)


def con_act(input_layers, n_filters, kernel=(1, 1), strides=1, activation='relu'):
    """

    :param input_layers:
    :param n_filters:
    :param kernel:
    :param strides:
    :param activation:
    :return:
    """
    con_layer = Conv2D(n_filters, kernel_size=kernel, kernel_initializer='he_normal', strides=strides)(input_layers)
    return Activation(activation)(con_layer)


def Attention_Refinment_Module(input_layers, num_filters):
    """

    :param input_layers:
    :param num_filters:
    :return:
    """
    init = input_layers
    # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # filters = init._keras_shape[channel_axis]
    # print("filters", filters)
    filters = num_filters
    gap_shape = (1, 1, filters)

    gap_layers = GlobalAveragePooling2D()(init)
    gap_layers = Reshape(gap_shape)(gap_layers)
    con_bn_act_layer = con_bn_act(gap_layers, n_filters=num_filters, kernel=(1, 1), activation="sigmoid")
    mul_layer = multiply([input_layers, con_bn_act_layer])
    return mul_layer


def Feature_Fusion_Module(sp_layers, cp_layers, n_classes):
    """

    :param sp_layers:
    :param cp_layers:
    :param n_classes:
    :return:
    """
    filters = n_classes
    gap_shape = (1, 1, filters)
    ffm = Concatenate(axis=3)([sp_layers, cp_layers])      # filters = 3328
    con = con_bn_act(ffm, n_classes, (3, 3), strides=2)
    gap_layers = GlobalAveragePooling2D()(con)       # 1x1 卷积层不需要Reshape？
    gap_layers = Reshape(gap_shape)(gap_layers)

    con_1 = con_act(gap_layers, n_filters=n_classes)

    con_2 = con_act(con_1, n_filters=n_classes)
    mul_layer = multiply([con, con_2])
    ffm_out = add([con, mul_layer])
    return ffm_out


def Spatial_path(inputs):
    """

    :param inputs:
    :return:
    """
    con_1 = con_bn_act(inputs, n_filters=64, kernel=(3, 3), strides=2)
    con_2 = con_bn_act(con_1, n_filters=128, kernel=(3, 3), strides=2)
    con_3 = con_bn_act(con_2, n_filters=256, kernel=(3, 3), strides=2)
    return con_3


# def Context_Path(layer_13, layer_14, gap):
#     """
#
#     :param layer_13: xception_13(19x19x1024) name = block13_sepconv2_bn
#     :param layer_14: xception_14(10x10x2048) name = block14_sepconv2_act
#     :param gap: xception_gap (1x1x2048) name = avg_pool
#     :return:
#     """
#     block1 = Attention_Refinment_Module(layer_13, num_filters=1024)
# xception_13(19x19x1024) name = block13_sepconv2_bn
#     block2 = Attention_Refinment_Module(layer_14, num_filters=2048)
# xception_14(10x10x2048) name = block14_sepconv2_act
#     gap = multiply([block2, gap])
#     upsample_1 = UpSampling2D(size=(2, 2))(gap)


def Context_Path(layer_13, layer_14):
    """

    :param layer_13:
    :param layer_14:
    :return:
    """
    globalmax = GlobalAveragePooling2D()

    block1 = Attention_Refinment_Module(layer_13, num_filters=1024)
    block2 = Attention_Refinment_Module(layer_14, num_filters=2048)

    global_channels = globalmax(block2)
    block2_scaled = multiply([global_channels, block2])

    block1 = UpSampling2D(size=(4, 4), interpolation='bilinear')(block1)
    block2_scaled = UpSampling2D(size=(4, 4), interpolation='bilinear')(block2_scaled)

    cnc = Concatenate(axis=-1)([block1, block2_scaled])
    out = Cropping2D(cropping=((1, 0), (1, 0)))(cnc)

    return out


def final_model(inputs, layer_13, layer_14):
    """

    :param inputs:
    :param layer_13:
    :param layer_14:
    :return:
    """
    sp_out = Spatial_path(inputs)
    cp_out = Context_Path(layer_13, layer_14)
    fusion_out = Feature_Fusion_Module(cp_out, sp_out, 32)
    out_layer = UpSampling2D(size=(8, 8), interpolation='bilinear')(fusion_out)
    return out_layer


def bisenet_model():
    """

    :return:
    """
    inputs = Input(shape=Input_size)
    x = Lambda(lambda image: preprocess_input(image))(inputs)
    xception = Xception(weights='imagenet', input_shape=Input_size, include_top=False)

    tail_prev = xception.get_layer('block13_pool').output
    tail = xception.output

    output = final_model(x, tail_prev, tail)
    # inputs, xception_inputs, ans = get_model()
    model = Model(inputs=[inputs, xception.input], output=[output])
    return model


# 打印网络结构图验证：
def plot_model_(p_model):

    plot_model(p_model, to_file='./structure/bisenet_1.pdf', show_shapes=True)


if __name__ == "__main__":
    plot_model_(bisenet_model())
