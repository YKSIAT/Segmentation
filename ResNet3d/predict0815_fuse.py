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
from Resnet50SmoothNet.vis_utils import show_line_chart

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class CustomError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)   # 初始化父类
        self.errorinfo=ErrorInfo

    def __str__(self):
        return self.errorinfo


class PredictCase:
    def __init__(self):
        # self.case_path = case_path + "/*"
        self.patch_shape = [128, 128, 128]
        self.case_path = None
        self.save_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/test815_1/fuse_test_816"
        self.reference_ata_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTs19Mixture_N4_HM_Norm/" \
                                  r"Train/BraTS19_2013_11_1/BraTS19_2013_11_1_flair.nii.gz"

    @staticmethod
    def get_model_list(case_path):
        model_list = glob.glob(case_path)
        mask_name = None
        image_list = []
        for idx, name in enumerate(model_list):
            if name.split(".")[0].split("_")[-1] != "seg":
                image_list.append(name)
            elif name.split(".")[0].split("_")[-1] == "seg":
                mask_name = name
        return image_list, mask_name

    def get_patch_from_array(self, image_array):
        """

        :param image_array:
        :return:
        """
        shape = image_array.shape
        patch_shape = self.patch_shape
        Center_coordinate = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]
        # print("Center_coordinate", Center_coordinate)

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
            data.append(self.get_patch_from_array(nib.load(model_name).get_data()))

        fuse_data = self.fuse_data(data[0], data[1], data[2], data[3])
        mask_data = self.bin_label(self.get_patch_from_array(nib.load(seg_name).get_data()))
        return fuse_data, mask_data

    @staticmethod
    def nii_data_read_nib(data_path, header=True):
        """

        :param data_path: nii data path
        :param header: True or False
        :return: 3D array  # (H, W, D)
        """
        img = nib.load(data_path)
        image_array_data = img.get_data()
        if header:
            return image_array_data, img.affine, img.header
        else:
            return image_array_data

    def save_nib(self, img_data, filename):
        """

        :param img_data:
        :param image:
        :param filename:
        :return:
        """
        _, affine, _ = self.nii_data_read_nib(self.reference_ata_path)
        new_image = nib.Nifti1Image(img_data, affine=affine)
        nib.save(new_image, filename)

    @staticmethod
    def re_array(final_shape, center_array):
        """

        :param final_shape:
        :param center_array:
        :return:
        """
        shape = final_shape
        patch_shape = [128, 128, 128]
        final_array = np.zeros(final_shape)
        Center_coordinate = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]
        final_array[Center_coordinate[0] - int(patch_shape[0] / 2):Center_coordinate[0] + int(patch_shape[0] / 2),
                    Center_coordinate[1] - int(patch_shape[1] / 2):Center_coordinate[1] + int(patch_shape[1] / 2),
                    Center_coordinate[2] - int(patch_shape[2] / 2):Center_coordinate[2] + int(patch_shape[2] / 2)] \
            = center_array

        return final_array

    def processing(self, weights_path, r_path):
        """

        :param weights_path:
        :param r_path:
        :return:
        """
        case_path_list = glob.glob(r_path + "/*")
        num = len(case_path_list)
        print("case_num", num)

        input_layer, output = ResUNet()
        model = Model(inputs=input_layer, outputs=output)
        model.load_weights(weights_path)
        evaluation_function_wt = []
        evaluation_function_wt_01 = []
        average_wt = 0
        average_wt_01 = 0

        evaluation_function_tc = []
        evaluation_function_tc_01 = []
        average_tc = 0
        average_tc_01 = 0

        evaluation_function_et = []
        evaluation_function_et_01 = []
        average_et = 0
        average_et_01 = 0

        for index, case_name in enumerate(case_path_list):
            print(index, case_name)
            create_case_path = os.path.join(self.save_path, case_name.split("/")[-1])
            print("create_case_path", create_case_path)
            case_name = case_name + "/*"
            one_case_model_ab_path_list, one_case_seg_ab_path_list = self.get_model_list(case_name)
            # print(one_case_model_ab_path_list)  # t2, t1, flair, t1ce
            # print(one_case_seg_ab_path_list)
            one_case_fuse_data, one_case_mask_data = \
                self._get_data(one_case_model_ab_path_list, one_case_seg_ab_path_list)
            # (1, 128, 128, 128, 4) (128, 128, 128, 3)
            # print(one_case_fuse_data.shape, one_case_mask_data.shape)
            pre_mask = model.predict(one_case_fuse_data, batch_size=1)  # (1, 128, 128, 128, 3)
            # print("pre_mask.shape", pre_mask.shape)

            dice_coefficient_c_wt, dice_coefficient_01_c_wt = \
                calculate_metrics_dice(pre_mask[0, :, :, :, 0], one_case_mask_data[:, :, :, 0], optimal_threshold=0.55)
            average_wt += dice_coefficient_c_wt
            average_wt_01 += dice_coefficient_01_c_wt
            evaluation_function_wt.append(dice_coefficient_c_wt)
            evaluation_function_wt_01.append(dice_coefficient_01_c_wt)
            # print("dice_coefficient_c_wt", dice_coefficient_c_wt)
            # print("dice_coefficient_01_c_wt", dice_coefficient_01_c_wt)

            dice_coefficient_c_tc, dice_coefficient_01_c_tc = \
                calculate_metrics_dice(pre_mask[0, :, :, :, 1], one_case_mask_data[:, :, :, 1], optimal_threshold=0.5)
            evaluation_function_tc.append(dice_coefficient_c_tc)
            evaluation_function_tc_01.append(dice_coefficient_01_c_tc)
            average_tc += dice_coefficient_c_tc
            average_tc_01 += dice_coefficient_01_c_tc
            # print("dice_coefficient_c_tc", dice_coefficient_c_tc)
            # print("dice_coefficient_01_c_tc", dice_coefficient_01_c_tc)

            dice_coefficient_c_et, dice_coefficient_01_c_et = \
                calculate_metrics_dice(pre_mask[0, :, :, :, 2], one_case_mask_data[:, :, :, 2], optimal_threshold=0.355)
            evaluation_function_et.append(dice_coefficient_c_et)
            evaluation_function_et_01.append(dice_coefficient_01_c_et)
            average_et += dice_coefficient_c_et
            average_et_01 += dice_coefficient_01_c_et
            # print("dice_coefficient_c_et", dice_coefficient_c_et)
            # print("dice_coefficient_01_c_et", dice_coefficient_01_c_et)

            # for idx in range(pre_mask.shape[3]):
            #     print(idx)
            #     pre_image_0 = pre_mask[0, :, :, idx, 0]
            #     pre_image_0_final = one_case_mask_data[:, :, idx, 0]
            #     if idx > 35:
            #         show_img_multiplex_cutoff(pre_image_0, pre_image_0_final, cutoff=0.55, img_title="wt")
            #         sleep(1)
            #         show_img_multiplex_cutoff\
            #             (one_case_fuse_data[0, :, :, idx, 0], pre_image_0_final, cutoff=0.55, img_title="wt")
            #         sleep(1)
            #     pass
            # final = self.re_array((240, 240, 155), pre_mask[0, :, :, :, 0])
            # print("final shape", final.shape)
            # for idx in range(final.shape[2]):
            #     pre_image_0 = final[:, :, index]
            #     pre_image_0_final = one_case_mask_data[:, :, idx, 0]
            #     if idx > 35:
            #         show_img_multiplex_cutoff(pre_image_0, pre_image_0_final, cutoff=0.55, img_title="wt")
            #         sleep(1)
            #         # show_img_multiplex_cutoff\
            #         #     (one_case_fuse_data[0, :, :, idx, 0], pre_image_0_final, cutoff=0.55, img_title="wt")
            #         # sleep(1)

            # create_case_path = os.path.join(self.save_path, case_name.split("/")[-1])
            # filename = create_case_path + "_seg.nii.gz"
            # print("filename", filename)
            # self.save_nib(pre_mask[0, :, :, :, 0], filename)

            # final = self.re_array((240, 240, 155), pre_mask[0, :, :, :, 0])
            # filename_complete = create_case_path + "_complete_c" + "_seg.nii.gz"
            # print("filename_complete", filename_complete)
            # self.save_nib(final, filename_complete)

            flair_data = one_case_fuse_data[0, :, :, :, 0]
            filename_complete_flair = create_case_path + "_complete_c" + "_flair.nii.gz"
            self.save_nib(flair_data, filename_complete_flair)














        # show_line_chart(list(range(len(evaluation_function_wt))), evaluation_function_wt, evaluation_function_wt_01,
        #                 image_name="The results of wt")
        # show_line_chart(list(range(len(evaluation_function_wt))), evaluation_function_tc, evaluation_function_tc_01,
        #                 image_name="The results of tc")
        # show_line_chart(list(range(len(evaluation_function_wt))), evaluation_function_et, evaluation_function_et_01,
        #                 image_name="The results of et")
        #
        # print("wt", average_wt/num)
        # print("wt_01", average_wt_01/num)
        #
        # print("tc", average_tc/num)
        # print("tc_01", average_tc_01/num)
        #
        # print("et", average_et/num)
        # print("et_01", average_et_01/num)



if __name__ == "__main__":
    path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/test815_1/Test815"
    w_p = r"./checkpoint_813/model3d_813_01.h5"
    prediction = PredictCase()
    prediction.processing(w_p, path)
































