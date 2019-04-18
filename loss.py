from keras.backend import sum, flatten
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import numpy as np
from keras.layers import *
from medpy import metric
from data_preprocessing_3 import *
smooth = 0.001


def dice_coefficient(y_true, y_pred):
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) + smooth)


def multiplex_dice_coefficient(y_true, y_pred):
    """

    :param y_true: (240, 240, 3)
    :param y_pred:
    :return:
    """
    pass
    # label : 1, 2, 4
    shape = y_true.shape
    print("shape", shape)
    # true_slices = list()
    # pred_slices = list()
    multiplex_dice_value = list()
    for idx in range(3):
        # true_slices.append(y_true[:, :, idx])
        # pred_slices.append(y_pred[:, :, idx])
        multiplex_dice_value.append(dice_coefficient(y_true[:, :, idx], y_pred[:, :, idx]))
    sum_multiplex_dice_value = multiplex_dice_value[0] + multiplex_dice_value[1] + multiplex_dice_value[2]
    return sum_multiplex_dice_value


def dice_coefficient_loss(y_true, y_pred):
    return 1. - dice_coefficient(y_true, y_pred)


def multiplex_dice_coefficient_loss(y_true, y_pred):
    multiplex_loss = 1. - multiplex_dice_coefficient(y_true, y_pred)
    return multiplex_loss


def plot_metrics_multiplex_dice(y_true, y_pred):
    shape = y_true.shape
    multiplex_dice_value = list()
    for idx in range(shape[2]):
        multiplex_dice_value.append(dice_coefficient(y_true[:, :, idx], y_pred[:, :, idx]))
    return multiplex_dice_value[0], multiplex_dice_value[1], multiplex_dice_value[2]


def sensitivity(y_true, y_pred):
    # y_true_f = flatten(y_true)
    # y_pred_f = flatten(y_pred)
    # intersection = sum(y_true_f * y_pred_f)
    intersection = sum(y_true * y_pred).sum()
    return (intersection + smooth) / ((y_true * y_true).sum() + smooth)


def misc_measures(true_arr, pred_arr):
    cm = confusion_matrix(true_arr.flatten(), pred_arr.flatten())
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sens = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    spec = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    return acc, sens, spec


# def specificity(y_true, y_pred):
#     y_true_f = flatten(y_true)
#     y_pred_f = flatten(y_pred)
#     intersection = sum(y_true_f * y_pred_f)
#     return (intersection + smooth) / (sum(y_pred * y_pred) + smooth)


def binary_dice3d(s, g):
    # dice score of two 3D volumes
    num = np.sum(np.multiply(s, g))
    denominator = s.sum() + g.sum()
    if denominator == 0:
        return 1
    else:
        return 2.0*num/denominator


def sensitivity_1(seg, ground):
    # computer false negative rate
    num = np.sum(np.multiply(ground, seg))
    denominator = np.sum(ground)
    if denominator == 0:
        return 1
    else:
        return num/denominator


def spec(seg, ground):
    # computes false positive rate
    num = np.sum(np.multiply(ground == 0, seg <= 0.02))
    denominator = np.sum(ground == 0)
    if denominator == 0:
        return 1
    else:
        return num/denominator


def show_img_multiplex(predict_mask, true_mask):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("predict_mask")
    plt.imshow(predict_mask, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("true_mask")
    plt.imshow(true_mask, cmap='gray')
    plt.show()


def sensitivity_2(y_true, y_pred):
    smooth = 0.0001
    # y_true_f = flatten(y_true)
    # y_pred_f = flatten(y_pred)
    # intersection = sum(y_true_f * y_pred_f)
    intersection = (y_true * (y_pred >= 0.23)).sum()
    return (intersection + smooth) / ((y_true * y_true).sum() + smooth)


def spec_2(y_pred, y_true):
    smooth = 0.001
    # computes false positive rate
    # num = np.sum(np.multiply(ground == 0, seg <= 0.02))
    intersection = ((y_true == 0) * (y_pred <= 0.23)).sum()
    denominator = (y_true == 0).sum()
    # show_img_multiplex((y_true == 0))
    # if denominator == 0:
    #     return 1
    # else:
    return (intersection + smooth) / ((denominator * denominator).sum() + smooth)


# def calculate_metrics(pred, target):
#     sens = metric.sensitivity(pred, target)
#     spec = metric.specificity(pred, target)
#     dice = metric.dc(pred, target)
#     return sens, spec, dice

def calculate_cut_off(predict_mask, true_mask):
    fpr, tpr, thresholds = roc_curve(true_mask.flatten(), predict_mask.flatten())
    roc_auc = auc(fpr, tpr)
    print("roc_auc", roc_auc)
    print("thresholds", thresholds)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def calculate_metrics(predict_mask, true_mask, optimal_threshold):
    smooth = 0.0001
    confusion = confusion_matrix(true_mask.flatten(), (predict_mask >= optimal_threshold).flatten())
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity__ = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity__ = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity__))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))

    intersection = (true_mask * predict_mask).sum()
    dice_coefficient_ = (2. * intersection + smooth) / \
                        ((true_mask * true_mask).sum() + (predict_mask * predict_mask).sum() + smooth)

    return specificity, sensitivity__, dice_coefficient_



