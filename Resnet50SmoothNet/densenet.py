from keras.layers import Input, ZeroPadding2D, Conv2D, Dense,\
    GlobalMaxPooling2D, BatchNormalization, Activation, MaxPooling2D, Concatenate, \
    AveragePooling2D, GlobalAveragePooling2D, Reshape, multiply, UpSampling2D, Add
from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras.utils.vis_utils import plot_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def DenseNet(blocks, include_top=False, weights_path=None, input_tensor=None,
             input_shape=None, pooling="avg", classes=1000, **kwargs):

    """
    Instantiates the DenseNet architecture.
    :param blocks: numbers of building blocks for the four dense layers.
    :param include_top: include_top: whether to include the fully-connected
            layer at the top of the network.
    :param weights_path: the path to the weights file to be loaded.
    :param input_tensor: optional Keras tensor
    :param input_shape:
    :param pooling:
    :param classes:
    :param kwargs:
    :return: A Keras model instance.

    """

    # global backend, layers, models, keras_utils
    # if not (kernel_initializer in {'he_normal', None} or os.path.exists(weights_path)):
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization), `imagenet` '
    #                      '(pre-training on ImageNet), '
    #                      'or the path to the weights file to be loaded.')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            print(not K.is_keras_tensor(input_tensor))
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1

    # print(bn_axis)
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv', kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)

    n_filters = x._keras_shape[bn_axis]
    re_shape = (1, 1, n_filters)
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Reshape(re_shape, name='gap_layer')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)
            x = Reshape(re_shape, name='gap_layer')(x)

    if blocks == [6, 12, 24, 16]:
        model = Model(img_input, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(img_input, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(img_input, x, name='densenet201')
    else:
        model = Model(inputs=img_input, outputs=x, name='densenet')
    # if weights_path is not None:
    #     if os.path.exists(weights_path):
    #         raise ValueError('The path to the weights file to be loaded is nonexistent.')
    # else:
    #     model.load_weights(weights_path)
    return model


def dense_block(x, blocks, name):
    """
    A dense block. BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv
    :param x: input tensor.
    :param blocks: integer, the number of building blocks.
    :param name: string, block label.
    :return: output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def conv_block(x, growth_rate, name):
    """
    A building block for a dense block.
    :param x: input tensor.
    :param growth_rate: float, growth rate at dense layers.
    :param name: string, block label.
    :return: Output tensor for the block.
    """
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv', kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(0.01))(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding="same", use_bias=False, name=name + '_2_conv', kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(0.01))(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def transition_block(x, reduction, name):
    """
    "BN+ReLU+1x1 Conv+2x2 AvgPooling"
    :param x: input tensor.
    :param reduction: float, compression rate at transition layers.
    :param name:string, block label.
    :return:  output tensor for the block.
    """
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.01), name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def DenseNet121(include_top=False,
                weights_path=None,
                input_tensor=None,
                input_shape=None,
                pooling="avg",
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights_path,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)


def DenseNet169(include_top=False,
                weights_path=None,
                input_tensor=None,
                input_shape=None,
                pooling="avg",
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights_path,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)


def DenseNet201(include_top=False,
                weights_path=None,
                input_tensor=None,
                input_shape=None,
                pooling="avg",
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights_path,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)


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


def Attention_Refinment_Module(input_layer, num_filters):
    """

    :param input_layer:
    :param num_filters:
    :return:
    """
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    filters = x._keras_shape[bn_axis]
    gap_shape = (1, 1, filters)

    gap_layers = GlobalAveragePooling2D()(input_layer)
    gap_layers = Reshape(gap_shape)(gap_layers)
    con_bn_act_layer = con_bn_act(gap_layers, n_filters=filters, kernel=(1, 1), activation="sigmoid")
    mul_layer = multiply([input_layer, con_bn_act_layer])
    return mul_layer


def test_arm():
    input = Input(shape=(32, 32, 512))
    output = Attention_Refinment_Module(input, 512)
    model = Model(input, output)
    return model
    pass


def ResPFN(input_shape=(240, 240, 12)):
    """

    :param input_shape:
    :param fuse_flge:
    :return:
    """
    model = DenseNet121(input_shape=input_shape)
    input = model.input
    dense1 = model.get_layer("conv2_block6_concat").output    # (None, 60, 60, 256)
    dense2 = model.get_layer("conv3_block12_concat").output     # (None, 30, 30, 1024)
    dense3 = model.get_layer("conv4_block24_concat").output     # (None, 15, 15, 1024)
    dense4 = model.get_layer("conv5_block16_concat").output     # (None, 7, 7, 1024)
    gap = model.get_layer("gap_layer").output    # (None, 1, 1, 1024)

    gap_up = UpSampling2D(size=(7, 7), data_format=None, interpolation='nearest', name="gap_up")(gap)
    gap_up = Conv2D(1024, 1, use_bias=False, activation='relu', name="gap_up_1024", kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(0.01))(gap_up)
    dense4_f = Conv2D(1024, 1, activation='relu', name="conv_pool5", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(0.01))(dense4)
    feature_1 = Add()([gap_up, dense4_f])     # shape=(?, 7, 7, 1024)

    feature_1_up = Conv2D(512, 1, activation='relu', name="feature_1_up", kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(0.01))\
        (ZeroPadding2D(((1, 0), (1, 0)))(UpSampling2D(size=(2, 2))(feature_1)))
    dense3_f = Conv2D(512, 1, activation='relu', name="dense3_f", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(0.01))(dense3)
    feature_2 = Add()([feature_1_up, dense3_f])

    feature_2_up = Conv2D(256, 1, activation='relu', name="feature_2_up", kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size=(2, 2))(feature_2))
    dense2_f = Conv2D(256, 1, activation='relu', name="dense2_f", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(0.01))(dense2)
    feature_3 = Add()([feature_2_up, dense2_f])

    feature_3_up = Conv2D(128, 1, activation='relu', name="feature_3_up", kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size=(2, 2))(feature_3))
    dense1_f = Conv2D(128, 1, activation='relu', name="dense1_f", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(0.01))(dense1)
    feature_4 = Add()([feature_3_up, dense1_f])
    out = Conv2D(1, 1, activation="sigmoid", name="out")(UpSampling2D(size=(4, 4))(feature_4))

    model_d = Model(inputs=input, outputs=out)
    return model_d

    # Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal',
    #        kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size=(2, 2))(conv7))
    #
    # feature_1 = Conv2D(1024, 1, use_bias=False, activation='relu', name="gap_up_1024", kernel_initializer="he_normal",
    #                    kernel_regularizer=regularizers.l2(0.01))(pool4)
    #
    #
    # extract_feature1=""

    # model1 = Model(input, feature_1_up)
    # return(model1)
    pass


if __name__ == "__main__":
    # blocks = [6, 12, 32, 32]
    model = ResPFN()
    print(model.output)
    plot_model(model, to_file='./model_ResPFN.pdf', show_shapes=True)

















