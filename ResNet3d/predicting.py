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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class CustomError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)   # 初始化父类
        self.errorinfo=ErrorInfo

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

        assert (Center_coordinate[0] + int(patch_shape[0] / 2) > shape[0]) or \
               (Center_coordinate[0] - int(patch_shape[0] / 2) > 0), "Out of size range"
        assert (Center_coordinate[1] + int(patch_shape[1] / 2) > shape[1]) or \
               (Center_coordinate[1] - int(patch_shape[1] / 2) > 0), "Out of size range"
        assert (Center_coordinate[2] + int(patch_shape[2] / 2) > shape[0]) or \
               (Center_coordinate[2] - int(patch_shape[2] / 2) > 0), "Out of size range"

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
    def bin_label(label_data,  region_type="whole", all_labels=True):
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
            print("fuse_mask.shape", fuse_mask.shape)      # fuse_mask.shape (128, 128, 128, 3)
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

    def _get_data(self, model_list, seg_name, location_type,
                  patch_location_wh_plane=None,
                  patch_location_d_dimensionality=None):
        """

        :param model_list:
        :param seg_name:
        :return:
        """
        assert location_type in ["center", "branch"]
        data = []

        if location_type == "center":
            for idx, model_name in enumerate(model_list):
                data.append(self.GetPatchFromArray(nib.load(model_name).get_data()))

            fuse_data = self.fuse_data(data[0], data[1], data[2], data[3])
            mask_data = self.bin_label(self.GetPatchFromArray(nib.load(seg_name).get_data()))
            return fuse_data, mask_data

        elif location_type == "branch":
            for idx, model_name in enumerate(model_list):
                data.append\
                    (get_patch_from_array_around_ranch(nib.load(model_name).get_data(),
                                                       patch_location_wh_plane=patch_location_wh_plane,
                                                       patch_location_d_dimensionality=patch_location_d_dimensionality)
                     )

            fuse_data = self.fuse_data(data[0], data[1], data[2], data[3])
            mask_data = get_patch_from_array_around_ranch(nib.load(seg_name).get_data(),
                                                          patch_location_wh_plane=patch_location_wh_plane,
                                                          patch_location_d_dimensionality=patch_location_d_dimensionality)
            mask_data = self.bin_label(mask_data)

            return fuse_data, mask_data

    def processing(self, weights_path):
        """

        :param weights_path:
        :return:
        """

        input_layer, output = ResUNet()
        model = Model(inputs=input_layer, outputs=output)
        model.load_weights(weights_path)

        model_list, seg_name = self.get_model_list()
        fuse_data_c, mask_data_c = self._get_data(model_list, seg_name, location_type="center")
        # fuse_data_shape (1, 128, 128, 128, 4)  mask_data_shape (128, 128, 128, 3)
        fuse_data_00, mask_data_00 = self._get_data(model_list, seg_name, location_type="branch",
                                                    patch_location_wh_plane="00",
                                                    patch_location_d_dimensionality="front")

        fuse_data_01, mask_data_01 = self._get_data(model_list, seg_name, location_type="branch",
                                                    patch_location_wh_plane="01",
                                                    patch_location_d_dimensionality="front")
        fuse_data_02, mask_data_02 = self._get_data(model_list, seg_name, location_type="branch",
                                                    patch_location_wh_plane="02",
                                                    patch_location_d_dimensionality="front")
        fuse_data_03, mask_data_03 = self._get_data(model_list, seg_name, location_type="branch",
                                                    patch_location_wh_plane="03",
                                                    patch_location_d_dimensionality="front")
        fuse_data_04, mask_data_04 = self._get_data(model_list, seg_name, location_type="branch",
                                                    patch_location_wh_plane="00",
                                                    patch_location_d_dimensionality="back")
        fuse_data_05, mask_data_05 = self._get_data(model_list, seg_name, location_type="branch",
                                                    patch_location_wh_plane="01",
                                                    patch_location_d_dimensionality="back")
        fuse_data_06, mask_data_06 = self._get_data(model_list, seg_name, location_type="branch",
                                                    patch_location_wh_plane="02",
                                                    patch_location_d_dimensionality="back")
        fuse_data_07, mask_data_07 = self._get_data(model_list, seg_name, location_type="branch",
                                                    patch_location_wh_plane="03",
                                                    patch_location_d_dimensionality="back")
        # print("fuse_data_00, mask_data_00", fuse_data_00.shape, mask_data_00.shape)
        # (1, 128, 128, 128, 4) (128, 128, 128, 3)

        true_mask_wt = fuse_array2complete_matrix(mask_data_00[:, :, :, 0], mask_data_01[:, :, :, 0],
                                                  mask_data_02[:, :, :, 0], mask_data_03[:, :, :, 0],
                                                  mask_data_04[:, :, :, 0], mask_data_05[:, :, :, 0],
                                                  mask_data_06[:, :, :, 0], mask_data_06[:, :, :, 0],
                                                  mask_data_c[:, :, :, 0])

        true_mask_tc = fuse_array2complete_matrix(mask_data_00[:, :, :, 1], mask_data_01[:, :, :, 1],
                                                  mask_data_02[:, :, :, 1], mask_data_03[:, :, :, 1],
                                                  mask_data_04[:, :, :, 1], mask_data_05[:, :, :, 1],
                                                  mask_data_06[:, :, :, 1], mask_data_06[:, :, :, 1],
                                                  mask_data_c[:, :, :, 1])

        true_mask_et = fuse_array2complete_matrix(mask_data_00[:, :, :, 2], mask_data_01[:, :, :, 2],
                                                  mask_data_02[:, :, :, 2], mask_data_03[:, :, :, 2],
                                                  mask_data_04[:, :, :, 2], mask_data_05[:, :, :, 2],
                                                  mask_data_06[:, :, :, 2], mask_data_06[:, :, :, 2],
                                                  mask_data_c[:, :, :, 2])

        print("true_mask_shape", true_mask_wt.shape)

        pre_mask_c = model.predict(fuse_data_c, batch_size=1)  # pre_mask_shape  (1, 128, 128, 128, 3)
        pre_mask_00 = model.predict(fuse_data_00, batch_size=1)
        pre_mask_01 = model.predict(fuse_data_01, batch_size=1)
        pre_mask_02 = model.predict(fuse_data_02, batch_size=1)
        pre_mask_03 = model.predict(fuse_data_03, batch_size=1)
        pre_mask_04 = model.predict(fuse_data_04, batch_size=1)
        pre_mask_05 = model.predict(fuse_data_05, batch_size=1)
        pre_mask_06 = model.predict(fuse_data_06, batch_size=1)
        pre_mask_07 = model.predict(fuse_data_07, batch_size=1)

        pre_mask_wt = fuse_array2complete_matrix(pre_mask_00[0, :, :, :, 0], pre_mask_01[0, :, :, :, 0],
                                                 pre_mask_02[0, :, :, :, 0], pre_mask_03[0, :, :, :, 0],
                                                 pre_mask_04[0, :, :, :, 0], pre_mask_05[0, :, :, :, 0],
                                                 pre_mask_06[0, :, :, :, 0], pre_mask_07[0, :, :, :, 0],
                                                 pre_mask_c[0, :, :, :, 0])

        pre_mask_tc = fuse_array2complete_matrix(pre_mask_00[0, :, :, :, 1], pre_mask_01[0, :, :, :, 1],
                                                 pre_mask_02[0, :, :, :, 1], pre_mask_03[0, :, :, :, 1],
                                                 pre_mask_04[0, :, :, :, 1], pre_mask_05[0, :, :, :, 1],
                                                 pre_mask_06[0, :, :, :, 1], pre_mask_07[0, :, :, :, 1],
                                                 pre_mask_c[0, :, :, :, 1])

        pre_mask_et = fuse_array2complete_matrix(pre_mask_00[0, :, :, :, 2], pre_mask_01[0, :, :, :, 2],
                                                 pre_mask_02[0, :, :, :, 2], pre_mask_03[0, :, :, :, 2],
                                                 pre_mask_04[0, :, :, :, 2], pre_mask_05[0, :, :, :, 2],
                                                 pre_mask_06[0, :, :, :, 2], pre_mask_07[0, :, :, :, 2],
                                                 pre_mask_c[0, :, :, :, 2])

        print("pre_mask_wt_shape", pre_mask_wt.shape)

        # print("pre_mask.shape", pre_mask.shape)
        # self.dice(pre_mask[0, :, :, :, 0], mask_data[0, :, :, :, 0])
        dice_coefficient_wt, dice_coefficient_01_wt = \
            calculate_metrics_dice(pre_mask_wt, true_mask_wt, optimal_threshold=0.55)
        print("dice_coefficient_wt is ", dice_coefficient_wt)
        print("dice_coefficient_01_wt is ", dice_coefficient_01_wt)

        dice_coefficient_tc, dice_coefficient_01_tc = \
            calculate_metrics_dice(pre_mask_tc, true_mask_tc, optimal_threshold=0.5)
        print("dice_coefficient_tc is ", dice_coefficient_tc)
        print("dice_coefficient_01_tc is ", dice_coefficient_01_tc)

        dice_coefficient_et, dice_coefficient_01_et = \
            calculate_metrics_dice(pre_mask_et, true_mask_et, optimal_threshold=0.35)
        print("dice_coefficient_et is ", dice_coefficient_et)
        print("dice_coefficient_01_et is ", dice_coefficient_01_et)

        dice_coefficient_c_wt, dice_coefficient_01_c_wt = \
            calculate_metrics_dice(pre_mask_c[0, :, :, :, 0], mask_data_c[:, :, :, 0], optimal_threshold=0.55)
        print("dice_coefficient_c_wt is ", dice_coefficient_c_wt)
        print("dice_coefficient_01_c_wt is ", dice_coefficient_01_c_wt)



        # print(pre_mask[0, :, :, :, 0].shape)
        # print(pre_mask[0, :, :, :, 1].shape)
        # print(pre_mask[0, :, :, :, 2].shape)

        for index in range(pre_mask_c.shape[3]):
            pre_image_0 = pre_mask_c[0, :, :, index, 0]
            pre_image_0_final = pre_mask_wt[:, :, index]

            pre_image_1 = pre_mask_c[0, :, :, index, 1]
            pre_image_1_final = pre_mask_tc[:, :, index]

            pre_image_2 = pre_mask_c[0, :, :, index, 2]
            pre_image_2_final = pre_mask_et[:, :, index]
            # image_image = fuse_data_c[0, :, :, index, 0]

            mask_slice_0 = mask_data_c[:, :, index, 0]
            mask_slice_0_final = true_mask_wt[:, :, index]

            mask_slice_1 = mask_data_c[:, :, index, 1]
            mask_slice_1_final = true_mask_tc[:, :, index]

            mask_slice_2 = mask_data_c[:, :, index, 2]
            mask_slice_2_final = true_mask_et[:, :, index]
            # print(index, pre_image.sum())
            if index > 35:
                show_img_multiplex_cutoff(pre_image_0, mask_slice_0, cutoff=0.55, img_title="wt")
                sleep(1)
                show_img_multiplex_cutoff(pre_image_0_final, mask_slice_0_final, cutoff=0.55, img_title="wt")
                sleep(1)
                # show_img_multiplex(mask_slice_0, image_image)
                # sleep(1)
                show_img_multiplex_cutoff(pre_image_1, mask_slice_1, cutoff=0.55, img_title="tc")
                sleep(1)
                show_img_multiplex_cutoff(pre_image_1_final, mask_slice_1_final, cutoff=0.55, img_title="wt")
                sleep(1)
                # show_img_multiplex(mask_slice_1, mask_slice_0)
                # sleep(1)
                show_img_multiplex_cutoff(pre_image_2, mask_slice_2, cutoff=0.35, img_title="et")
                sleep(1)
                show_img_multiplex_cutoff(pre_image_2_final, mask_slice_2_final, cutoff=0.55, img_title="wt")
                sleep(1)
                # show_img_multiplex(mask_slice_2, image_image)
                # sleep(1)
        pass

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
    path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTs19Mixture_N4_HM_Norm/Train/BraTS19_2013_11_1"
    w_p = r"./checkpoint_813/model3d_813_01.h5"
    prediction = PredictCase(path)
    model_list_, mask_name_ = prediction.get_model_list()
    model_list, seg_name = prediction.get_model_list()
    prediction.processing(w_p)
    # prediction._test_get_data()
    print(model_list_)
    print(mask_name_)


















