from model_bn import *
from data_preprocessing_3 import *
from keras.layers import *
from sklearn import metrics
import os
import numpy as np
from PIL import Image
from keras.utils import multi_gpu_model


def dice_coefficient_(y_true, y_pred):
    smooth = 0.0001
    # y_true_f = y_true.flatten()
    # y_pred_f = y_pred.flatten()
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)


def show_img(ori_img):
    plt.imshow(ori_img[:, :], cmap='gray')  # channel_last(x, y, z)
    plt.show()


def evaluate():
    # model = multi_gpu_model(unet_scSE(), gpus=2)
    model = unet()
    model.load_weights("brats_se.h5")
    # model.load_weights('brats_se_2.h5')

    image_1 = np.expand_dims(np.load("/home/yk/Project/keras/jia/Brats18_TCIA04_361_1.npy"), axis=0)
    image_2 = np.expand_dims(np.load("/home/yk/Project/keras/jia/Brats18_TCIA03_375_1.npy"), axis=0)
    image_3 = np.expand_dims(np.load("/home/yk/Project/keras/jia/Brats18_CBICA_AWI_1.npy"), axis=0)
    image_4 = np.expand_dims(np.load("/home/yk/Project/keras/jia/ZD_234.npy"), axis=0)
    print(type(image_1))
    print(image_1.shape)


    counter = 0
    dice = 0
    data_ = list()
    data_.append(image_1)
    data_.append(image_2)
    data_.append(image_3)
    data_.append(image_4)
    print("data_[0].shape", data_[0].shape)
    for idx, value in enumerate(data_):
        predict_mask = model.predict(value, batch_size=1, verbose=0, steps=None)[0]
        predict_mask_ = predict_mask[:, :, 0]
        show_img(predict_mask_)
    # for index, img in enumerate(val_data):
    #     counter += 1
    #     predict_mask = model.predict(img, batch_size=1, verbose=0, steps=None)[0]
    #
    #     print("img.shape", img.shape)
    #     print("true_mask.shape", true_mask.shape)
    #     print("predict_mask.shape", predict_mask.shape)
    #     true_mask_ = true_mask[0, :, :]
    #     predict_mask_ = predict_mask[:, :, 0]
    #     # show_img_multiplex(predict_mask_, true_mask_)
    #     # dice_ = dice_coefficient_(true_mask_.astype(np.float32), predict_mask_.astype(np.float32))
    #     # dice = dice + dice_
    #     np.save("predict_mask_", predict_mask_)
    #     # statistics(predict_mask_)
    #     if counter == 2:
    #         ave_dice = dice/2
    #         # show_img_multiplex(predict_mask_, true_mask_)
    #         break

    return 0


if __name__ == "__main__":
    evaluate()


