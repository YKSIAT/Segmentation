import matplotlib.pyplot as plt
import os
from keras.utils import plot_model
import numpy as np
import keras
import cv2
import random


def show_img_multiplex(predict_mask, true_mask):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("predict_mask")
    plt.imshow(predict_mask, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("true_mask")
    plt.imshow(true_mask, cmap='gray')
    plt.show()


def show_img(ori_img):
    plt.imshow(ori_img[:, :], cmap='gray')  # channel_last(x, y, z)
    plt.show()


def plot_model_map(model, save_path, save_name):
    path = os.path.join(save_path, save_name)
    plot_model(model, to_file=path, show_shapes=True)


def show_image_color(mask_array, n_classes=3):
    """

    :param mask_array:
    :param n_classes:
    :return:
    """
    brain_lesion = [200 / 255, 0, 0]
    normal_region = [0, 0, 0]
    colors = [normal_region, brain_lesion, brain_lesion]
    # colors = [(random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)) for _ in range(n_classes)]
    shape = mask_array.shape
    seg_img = np.zeros((shape[0], shape[1], 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((mask_array[:, :] == c) * (colors[c][0])).astype('float16')
        seg_img[:, :, 1] += ((mask_array[:, :] == c) * (colors[c][1])).astype('float16')
        seg_img[:, :, 2] += ((mask_array[:, :] == c) * (colors[c][2])).astype('float16')

    return seg_img


def plot_fusion_image(image1, image2, image1_alpha, image2_alpha):
    """

    :param image1:
    :param image2:
    :param image1_alpha:
    :param image2_alpha:
    :return: 0
    """
    fig, ax = plt.subplots()
    ax.imshow(image1, cmap="gray", alpha=image1_alpha)
    ax.imshow(image2, cmap="gray", alpha=image2_alpha)
    plt.show()


def transform(predict_mask, optimal_threshold):
    return predict_mask >= optimal_threshold


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    image_path = "C:\\Users\\SIAT\\Desktop\\BtaTs2018\\true_mask.npy"
    mask = np.load(image_path)
    show = show_image_color(mask, n_classes=2)
    plt.imshow(show)
    plt.show()





