from keras.backend import sum, flatten
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import numpy as np
from keras.layers import *
from medpy import metric
import tensorflow as tf
from data_preprocessing import *
from keras import backend as K
smooth = 0.001
def binary_focal_loss(y_true, y_pred):
    """

    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a sigmoid
    :param gamma:
    :param alpha:
    :return: Output tensor.
    """
    gamma = 2.
    alpha = .25
    pt_1 = np.where(np.equal(y_true, 1), y_pred, np.ones_like(y_pred))
    pt_0 = np.where(np.equal(y_true, 0), y_pred, np.zeros_like(y_pred))

    epsilon = 1e-08
    # clip to prevent NaN's and Inf's
    pt_1 = np.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = np.clip(pt_0, epsilon, 1. - epsilon)

    return -np.sum(alpha * np.power(1. - pt_1, gamma) * np.log(pt_1), axis=-1) \
           - np.sum((1 - alpha) * np.power(pt_0, gamma) * np.log(1. - pt_0), axis=-1)


# def w_dice(y_true, y_pred):
#     """
#
#     :param y_true: A tensor of the same shape as `y_pred`
#     :param y_pred: A tensor resulting from a sigmoid
#     :param gamma:
#     :param alpha:
#     :return: Output tensor.
#     """
#     gamma = 2.
#     alpha = .25
#     pt_1 = np.where(np.equal(y_true, 1), y_pred, np.ones_like(y_pred))
#     pt_0 = np.where(np.equal(y_true, 0), y_pred, np.zeros_like(y_pred))
#     k_wt = ""
#
#     epsilon = 1e-08
#     # clip to prevent NaN's and Inf's
#     pt_1 = np.clip(pt_1, epsilon, 1. - epsilon)
#     pt_0 = np.clip(pt_0, epsilon, 1. - epsilon)
#
#     return -np.sum(alpha * np.power(1. - pt_1, gamma) * np.log(pt_1), axis=-1) \
#            - np.sum((1 - alpha) * np.power(pt_0, gamma) * np.log(1. - pt_0), axis=-1)


def weighted_dice_loss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_wt = y_true[:, :, :, 0]
    y_true_tc = y_true[:, :, :, 1]
    y_true_et = y_true[:, :, :, 2]

    y_pre_wt = y_pred[:, :, :, 0]
    y_pre_tc = y_pred[:, :, :, 1]
    y_pre_et = y_pred[:, :, :, 2]

    k_wt = K.sum(y_true_wt)
    k_tc = K.sum(y_true_tc)
    k_et = K.sum(y_true_et)

    loss = ""


def dice_coefficient(y_true, y_pred):
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) + smooth)





















































