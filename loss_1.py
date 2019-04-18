# from keras.backend import sum, flatten
import numpy as np
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import roc_curve, auc, confusion_matrix
from data_preprocessing_3 import *
smooth = 0.001


def dice_coefficient(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # y_true_f = flatten(y_true)
    # y_pred_f = flatten(y_pred)
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)
    # intersection = (true_mask * predict_mask).sum()
    # (2. * intersection + smooth) / ((true_mask * true_mask).sum() + (predict_mask * predict_mask).sum() + smooth)


def dice_coefficient_loss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    return 1. - dice_coefficient(y_true, y_pred)


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


def multiplex_dice_coefficient_loss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    multiplex_loss = 1. - multiplex_dice_coefficient(y_true, y_pred)
    return multiplex_loss


def plot_metrics_multiplex_dice(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    shape = y_true.shape
    multiplex_dice_value = list()
    for idx in range(shape[2]):
        multiplex_dice_value.append(dice_coefficient(y_true[:, :, idx], y_pred[:, :, idx]))
    return multiplex_dice_value[0], multiplex_dice_value[1], multiplex_dice_value[2]


def sensitivity(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    intersection = sum(y_true * y_pred).sum()
    return (intersection + smooth) / ((y_true * y_true).sum() + smooth)


# def misc_measures(true_arr, pred_arr):
#     cm = confusion_matrix(true_arr.flatten(), pred_arr.flatten())
#     acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
#     sens = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
#     spec = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
#     return acc, sens, spec


# def specificity(y_true, y_pred):
#     y_true_f = flatten(y_true)
#     y_pred_f = flatten(y_pred)
#     intersection = sum(y_true_f * y_pred_f)
#     return (intersection + smooth) / (sum(y_pred * y_pred) + smooth)


# def calculate_metrics(pred, target):
#     sens = metric.sensitivity(pred, target)
#     spec = metric.specificity(pred, target)
#     dice = metric.dc(pred, target)
#     return sens, spec, dice


def calculate_cut_off(predict_mask, true_mask):
    """

    :param predict_mask:
    :param true_mask:
    :return:
    """
    fpr, tpr, thresholds = roc_curve(true_mask.flatten(), predict_mask.flatten())
    roc_auc = auc(fpr, tpr)
    print("roc_auc", roc_auc)
    print("thresholds", thresholds)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def calculate_metrics(predict_mask, true_mask, optimal_threshold):
    """

    :param predict_mask:
    :param true_mask:
    :param optimal_threshold:
    :return:
    """
    smooth = 0.0001

    confusion = confusion_matrix(true_mask.flatten(), (predict_mask >= optimal_threshold).flatten())
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    # print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    # print("Specificity: " + str(specificity))
    sensitivity_ = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity_ = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    # print("Sensitivity: " + str(sensitivity_))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    # print("Precision: " + str(precision))

    # intersection = (true_mask * predict_mask).sum()
    # dice_coefficient_ = (2. * intersection + smooth) / \
    #                     ((true_mask * true_mask).sum() + (predict_mask * predict_mask).sum() + smooth)
    dice_coefficient_ = dice_coefficient(true_mask, predict_mask)
    hausdorff_distance_ = hausdorff_distance(predict_mask, true_mask, optimal_threshold)

    return specificity, sensitivity_, dice_coefficient_, hausdorff_distance_


def hausdorff_distance(predict_mask, true_mask, optimal_threshold):

    # quality = dict()
    # quality["Hausdorff"] = directed_hausdorff(predict_mask >= optimal_threshold, true_mask)[0]
    # labelPred=sitk.GetImageFromArray(lP, isVector=False)
    # labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    # hausdorff_computer = sitk.HausdorffDistanceImageFilter()
    # hausdorff_computer.Execute(true_mask, predict_mask >= optimal_threshold)
    # quality["avgHausdorff"] = hausdorff_computer.GetAverageHausdorffDistance()
    # quality["Hausdorff"] = hausdorff_computer.GetHausdorffDistance()
    #
    # dice_computer = sitk.LabelOverlapMeasuresImageFilter()
    # dice_computer.Execute(true_mask, predict_mask >= optimal_threshold)
    # quality["dice"] = dice_computer.GetDiceCoefficient()

    return directed_hausdorff(predict_mask >= optimal_threshold, true_mask)[0]
