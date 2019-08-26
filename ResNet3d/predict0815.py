import numpy as np
from glob import glob
import glob
import os
from ResNet3D.ResUNet3D import ResUNet
import nibabel as nib
from keras.models import Model
from utils import show_img_multiplex
from time import sleep
from Resnet50SmoothNet.vis_utils import show_img_multiplex_cutoff
from loss import calculate_metrics, calculate_metrics_dice
from ResNet3D.patch_utils import get_patch_from_array_around_ranch, fuse_array2complete_matrix
from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class CustomError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)  # 初始化父类
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo


class PredictCase:
    def __init__(self, case_path):
        self.case_path = case_path + "/*"
        self.patch_shape = [128, 128, 128]

        pass

    def get_model_list(self):
        model_list = glob.glob(self.case_path)
        mask_name = None
        image_list = []
        for idx, name in enumerate(model_list):

            if name.split(".")[0].split("_")[-1] != "seg":
                image_list.append(name)
            elif name.split(".")[0].split("_")[-1] == "seg":
                mask_name = name
        return image_list, mask_name

    def GetPatchFromArray(self, image_array):
        """

        :param image_array:
        :param p_shape:
        :return:
        """
        shape = image_array.shape
        patch_shape = self.patch_shape
        # Center_coordinate = [120, 120, 75]
        Center_coordinate = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]

        assert (Center_coordinate[0] + int(patch_shape[0] / 2) <= shape[0]), "Out of size range"
        assert (Center_coordinate[0] - int(patch_shape[0] / 2) >= 0), "Out of size range"

        assert (Center_coordinate[0] + int(patch_shape[1] / 2) <= shape[0]), "Out of size range"
        assert (Center_coordinate[0] - int(patch_shape[1] / 2) >= 0), "Out of size range"

        assert (Center_coordinate[0] + int(patch_shape[2] / 2) <= shape[0]), "Out of size range"
        assert (Center_coordinate[0] - int(patch_shape[2] / 2) >= 0), "Out of size range"

        patch_data = image_array[
                     Center_coordinate[0] - int(patch_shape[0] / 2):Center_coordinate[0] + int(patch_shape[0] / 2),
                     Center_coordinate[1] - int(patch_shape[1] / 2):Center_coordinate[1] + int(patch_shape[1] / 2),
                     Center_coordinate[2] - int(patch_shape[2] / 2):Center_coordinate[2] + int(patch_shape[2] / 2)]
        return patch_data

    @staticmethod
    def fuse_data(data1, data2, data3, data4):
        """

        :param data1: flair
        :param data2: t1
        :param data3: t1ce
        :param data4: t2
        :return:
        """
        fuse_data = []
        y1 = data1[..., np.newaxis]
        y2 = data2[..., np.newaxis]
        y3 = data3[..., np.newaxis]
        y4 = data4[..., np.newaxis]
        fuse_data.append(np.concatenate((y1, y2, y3, y4), axis=-1))
        return np.array(fuse_data)

    @staticmethod
    def bin_label(label_data, region_type="whole", all_labels=True):
        """

        :param label_data:
        :param region_type:
        :param all_labels:
        :return:
        """

        label_num = [1, 2, 4]
        fuse_mask = []
        label_data_shape = label_data.shape
        assert len(label_data_shape) == 3, "The shape of label data should be 3d"
        seg_labels = np.zeros((label_data_shape[0], label_data_shape[1], label_data_shape[2], len(label_num)))
        whole_mask = np.zeros((label_data_shape[0], label_data_shape[1], label_data_shape[2]), dtype=np.uint8)
        core_mask = np.zeros((label_data_shape[0], label_data_shape[1], label_data_shape[2]), dtype=np.uint8)
        active_mask = np.zeros((label_data_shape[0], label_data_shape[1], label_data_shape[2]), dtype=np.uint8)
        # label_data[:, :, :][label_data[:, :, :] == 4] = 3
        try:
            for idx in range(len(label_num)):
                seg_labels[:, :, :, idx] = (label_data == int(label_num[idx])).astype(int)
            whole_mask = seg_labels[:, :, :, 0] + seg_labels[:, :, :, 1] + seg_labels[:, :, :, 2]
            fuse_mask.append(whole_mask)
            core_mask = seg_labels[:, :, :, 0] + seg_labels[:, :, :, 2]
            fuse_mask.append(core_mask)
            active_mask = seg_labels[:, :, :, 2]
            fuse_mask.append(active_mask)
        except Exception as error:  # 捕获所有可能发生的异常
            print("ERROR：", error)
        finally:
            pass
        # three_stacked = np.dstack((whole_mask, core_mask, active_mask))
        # print("three_stacked.shape", three_stacked.shape)
        if all_labels:
            fuse_mask = np.transpose(np.array(fuse_mask), [1, 2, 3, 0])
            # print("fuse_mask.shape", fuse_mask.shape)  # fuse_mask.shape (128, 128, 128, 3)
            return fuse_mask

        else:
            if region_type == "whole":
                return whole_mask
            elif region_type == "core":
                return core_mask
            elif region_type == "active":
                return active_mask
            else:
                raise CustomError('Parameter values need to be selected from "whole", "core" and "active"')

    def _get_data(self, model_list, seg_name):
        """

        :param model_list:
        :param seg_name:
        :return:
        """
        data = []
        for idx, model_name in enumerate(model_list):
            data.append(self.GetPatchFromArray(nib.load(model_name).get_data()))

        fuse_data = self.fuse_data(data[0], data[1], data[2], data[3])
        mask_data = self.bin_label(self.GetPatchFromArray(nib.load(seg_name).get_data()))
        return fuse_data, mask_data

    def processing(self, weights_path):
        """

        :param weights_path:
        :return:
        """
        dic = []
        dic_01 = []
        input_layer, output = ResUNet()
        model = Model(inputs=input_layer, outputs=output)
        model.load_weights(weights_path)

        model_list, seg_name = self.get_model_list()
        fuse_data_c, mask_data_c = self._get_data(model_list, seg_name)
        # fuse_data_shape (1, 128, 128, 128, 4)  mask_data_shape (128, 128, 128, 3)
        # print("fuse_data_00, mask_data_00", fuse_data_00.shape, mask_data_00.shape)
        # (1, 128, 128, 128, 4) (128, 128, 128, 3)

        pre_mask_c = model.predict(fuse_data_c, batch_size=1)  # pre_mask_shape  (1, 128, 128, 128, 3)

        # print("pre_mask.shape", pre_mask.shape)
        # self.dice(pre_mask[0, :, :, :, 0], mask_data[0, :, :, :, 0])
        dice_coefficient_c_wt, dice_coefficient_01_c_wt = \
            calculate_metrics_dice(pre_mask_c[0, :, :, :, 0], mask_data_c[:, :, :, 0], optimal_threshold=0.55)
        dic.append(dice_coefficient_c_wt)
        dic_01.append(dice_coefficient_01_c_wt)
        print("dice_coefficient_c_wt is ", dice_coefficient_c_wt)
        print("dice_coefficient_01_c_wt is ", dice_coefficient_01_c_wt)

        dice_coefficient_c_tc, dice_coefficient_01_c_tc = \
            calculate_metrics_dice(pre_mask_c[0, :, :, :, 1], mask_data_c[:, :, :, 1], optimal_threshold=0.5)
        dic.append(dice_coefficient_c_tc)
        dic_01.append(dice_coefficient_01_c_tc)
        print("dice_coefficient_c_tc is ", dice_coefficient_c_tc)
        print("dice_coefficient_01_c_tc is ", dice_coefficient_01_c_tc)

        dice_coefficient_c_et, dice_coefficient_01_c_et = \
            calculate_metrics_dice(pre_mask_c[0, :, :, :, 2], mask_data_c[:, :, :, 2], optimal_threshold=0.35)
        dic.append(dice_coefficient_c_et)
        dic_01.append(dice_coefficient_01_c_et)
        print("dice_coefficient_c_et is ", dice_coefficient_c_et)
        print("dice_coefficient_01_c_et is ", dice_coefficient_01_c_et)

        return dic, dic_01

    def _test_get_data(self, model_list, seg_name):
        """

        :param model_list:
        :param seg_name:
        :return:
        """
        data = []
        for idx, model_name in enumerate(model_list):
            data.append(self.GetPatchFromArray(nib.load(model_name).get_data()))

        fuse_data = self.fuse_data(data[0], data[1], data[2], data[3])
        mask_data = self.bin_label(self.GetPatchFromArray(nib.load(seg_name).get_data()))
        return fuse_data, mask_data

    @staticmethod
    def dice(y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        :return:
        """
        smooth = 0.001
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        intersection = (y_true * y_pred).sum()
        return (2. * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)


if __name__ == "__main__":
    # path_list = glob.glob(r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTs19Mixture_N4_HM_Norm/Train/*")
    # path_list = glob.glob(r"/home/yk/Project/keras/dataset/BraTs19DataSet/Test815_1/*")
    path_list = glob.glob(r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTs19Mixture_N4_HM_Norm/Train/*")
    # print(path_list)
    case_num = len(path_list)
    # print(case_num)
    # w_p = r"./checkpoint_813/model3d_818_01.h5"
    w_p = r"./checkpoint_813/model3d_823_01.h5"

    d_wt = 0
    d_01_wt = 0
    dice_tc = 0
    dice_01_tc = 0
    dice_et = 0
    dice_01_et = 0
    for idx, name in enumerate(path_list):
        print(idx, name)
        prediction = PredictCase(name)
        model_list_, mask_name_ = prediction.get_model_list()
        dice_wt, dice_01_wt = prediction.processing(w_p)

        K.clear_session()
        d_wt += dice_wt[0]
        d_01_wt += dice_01_wt[0]

        dice_tc += dice_wt[1]
        dice_01_tc += dice_01_wt[1]

        dice_et += dice_wt[2]
        dice_01_et += dice_01_wt[2]

    print("wt", d_wt/case_num)
    print("wt_01", d_01_wt / case_num)

    print("tc", dice_tc / case_num)
    print("tc_01", dice_01_tc / case_num)

    print("et", dice_et / case_num)
    print("et_01", dice_01_et / case_num)




















