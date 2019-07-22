import numpy as np
from keras import Model, layers
from keras.applications import vgg16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, MaxPool2D, concatenate, UpSampling2D


def getSegmentationArr(seg, nClasses, input_height, input_width):
    """

    :param seg: image label
    :param nClasses: Classes + 1
    :param input_height: image high
    :param input_width: image weight
    :return: high * weight * 
    """
    seg_labels = np.zeros((input_height, input_width, nClasses))
    for c in range(nClasses):
        seg_labels[:, :, c] = (seg == c).astype(int)
    seg_labels = np.reshape(seg_labels, (-1, nClasses))
    return seg_labels


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """

    x = np.argmax(image, axis=-1)
    return x


def UNet(nClasses, input_height, input_width):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    vgg_streamlined = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input)
    assert isinstance(vgg_streamlined, Model)

    # 解码层
    o = UpSampling2D((2, 2))(vgg_streamlined.output)
    o = concatenate([vgg_streamlined.get_layer(
        name="block4_pool").output, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block3_pool").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block2_pool").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block1_pool").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    # UNet网络处理输入时进行了镜面放大2倍，所以最终的输入输出缩小了2倍
    # 此处直接上采样置原始大小
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = Conv2D(nClasses, (1, 1), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation("relu")(o)

    o = Reshape((-1, nClasses))(o)
    o = Activation("softmax")(o)

    model = Model(inputs=img_input, outputs=o)
    return model

if __name__ == "__main__":
    a = [[3, 3, 5],
         [3, 3, 5],
         [3, 3, 5]]
    a = np.asarray(a)

    h = a.shape[0]
    w = a.shape[1]
    aa = getSegmentationArr(a, 6, h, w)
    aaa = np.reshape(aa, (h, w, 6))
    print(reverse_one_hot(aaa))