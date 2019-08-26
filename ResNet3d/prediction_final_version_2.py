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
# from ResNet3D.patch_utils import get_patch_from_array_around_ranch

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
        self.patch_shape = [160, 160, 128]
        self.case_path = None
        # self.save_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/test815_1/fuse_test_816"
        # self.save_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/brate19_prediction/pre_818_train"
        self.save_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/brate19_prediction/pre_825_val"
        self.reference_ata_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTs19Mixture_N4_HM_Norm/" \
                                  r"Train/BraTS19_2013_11_1/BraTS19_2013_11_1_flair.nii.gz"

    @staticmethod
    def get_model_list(case_path):
        model_list = glob.glob(case_path)
        # mask_name = None
        image_list = []
        for idx, name in enumerate(model_list):
            # if name.split(".")[0].split("_")[-1] != "seg":
            image_list.append(name)
            # elif name.split(".")[0].split("_")[-1] == "seg":
            #     mask_name = name
        return image_list

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

    # def _get_data(self, model_list, seg_name):
    #     """
    #
    #     :param model_list:
    #     :param seg_name:
    #     :return:
    #     """
    #     data = []
    #     for idx, model_name in enumerate(model_list):
    #
    #         data.append(self.get_patch_from_array(nib.load(model_name).get_data()))
    #     fuse_data = self.fuse_data(data[0], data[1], data[2], data[3])
    #
    #     mask_data = self.bin_label(self.get_patch_from_array(nib.load(seg_name).get_data()))
    #     return fuse_data, mask_data

    def _get_data(self, model_list, seg_name):
        """

        :param model_list:
        :param seg_name:
        :return:
        """
        data = {}
        for idx, model_name in enumerate(model_list):
            if model_name.split("/")[-1].split(".") == "flair.nii.gz":
                data["flair"] = self.get_patch_from_array(nib.load(model_name).get_data())
            elif model_name.split("/")[-1].split(".") == "t1.nii.gz":
                data["t1"] = self.get_patch_from_array(nib.load(model_name).get_data())
            elif model_name.split("/")[-1].split(".") == "t1ce.nii.gz":
                data["t1ce"] = self.get_patch_from_array(nib.load(model_name).get_data())
            elif model_name.split("/")[-1].split(".") == "t2.nii.gz":
                data["t2"] = self.get_patch_from_array(nib.load(model_name).get_data())
        fuse_data = self.fuse_data(data["flair"], data["t1"], data["t1ce"], data["t2"])

        mask_data = self.bin_label(self.get_patch_from_array(nib.load(seg_name).get_data()))
        return fuse_data, mask_data

    def get_patch_x(self, model_list, location="00"):

        data = []
        for idx, model_name in enumerate(model_list):
            data.append(get_patch_from_array_around_ranch(nib.load(model_name).get_data(), location))
        fuse_data = self.fuse_data(data[0], data[1], data[2], data[3])
        return fuse_data


        pass

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
        patch_shape = [160, 160, 128]
        final_array = np.zeros(final_shape)
        Center_coordinate = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]
        final_array[Center_coordinate[0] - int(patch_shape[0] / 2):Center_coordinate[0] + int(patch_shape[0] / 2),
                    Center_coordinate[1] - int(patch_shape[1] / 2):Center_coordinate[1] + int(patch_shape[1] / 2),
                    Center_coordinate[2] - int(patch_shape[2] / 2):Center_coordinate[2] + int(patch_shape[2] / 2)] \
            = center_array
        # final_array[0:P_shape[0], shape[1] - P_shape[1]:shape[1], 0:P_shape[2]]=

        return final_array

    @staticmethod
    def re_array_x(final_shape, patch_x_array, patch_num):
        """

        :param final_shape:
        :param patch_x_array:
        :param patch_num:
        :return:
        """
        assert patch_num in ["00", "01", "02", "03", "04", "05", "06", "07", "cc"]
        shape = (240, 240, 155)  # image shape
        P_shape = (160, 160, 128)  # patch shape
        patch_shape = P_shape
        Center_coordinate = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]
        # Center_coordinate = (120, 120, 77)  # Center coordinate
        final_array = np.zeros(final_shape)

        if patch_num == "00":
            final_array[0:P_shape[0], shape[1] - P_shape[1]:shape[1], 0:P_shape[2]] = patch_x_array
            return final_array
        elif patch_num == "01":
            final_array[shape[0] - P_shape[0]:shape[0], shape[1] - P_shape[1]:shape[1], 0:P_shape[2]] = patch_x_array
            return final_array
        elif patch_num == "02":
            final_array[0:P_shape[0], 0:P_shape[1], 0:P_shape[2]] = patch_x_array
            return final_array
        elif patch_num == "03":
            final_array[shape[0] - P_shape[0]:shape[0], 0:P_shape[1], 0:P_shape[2]] = patch_x_array
            return final_array
        elif patch_num == "04":
            final_array[0:P_shape[0], shape[1] - P_shape[1]:shape[1], shape[2] - P_shape[2]:shape[2]] = patch_x_array
            return final_array
        elif patch_num == "05":
            final_array[shape[0] - P_shape[0]:shape[0], shape[1] - P_shape[1]:shape[1], shape[2] - P_shape[2]:shape[2]]\
                = patch_x_array
            return final_array
        elif patch_num == "06":
            final_array[0:P_shape[0], 0:P_shape[1], shape[2] - P_shape[2]:shape[2]] = patch_x_array
            return final_array
        elif patch_num == "07":
            final_array[shape[0] - P_shape[0]:shape[0], 0:P_shape[1], shape[2] - P_shape[2]:shape[2]] = patch_x_array
            return final_array
        elif patch_num == "cc":
            final_array[Center_coordinate[0] - int(patch_shape[0] / 2):Center_coordinate[0] + int(patch_shape[0] / 2),
                        Center_coordinate[1] - int(patch_shape[1] / 2):Center_coordinate[1] + int(patch_shape[1] / 2),
                        Center_coordinate[2] - int(patch_shape[2] / 2):Center_coordinate[2] + int(patch_shape[2] / 2)]\
                = patch_x_array
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

        for index, case_name in enumerate(case_path_list):
            print(index, case_name)
            create_case_path = os.path.join(self.save_path, case_name.split("/")[-1])
            print("create_case_path", create_case_path)
            case_name = case_name + "/*"
            one_case_model_ab_path_list = self.get_model_list(case_name)
            # print(one_case_model_ab_path_list)  # t2, t1, flair, t1ce
            # print(one_case_seg_ab_path_list)
            data_sets = []
            data_sets01 = []
            optimal_threshold = 0.55
            for index_, patch_num in enumerate(["00", "01", "02", "03", "04", "05", "06", "07", "cc"]):
                one_case_fuse_data = \
                    self.get_patch_x(one_case_model_ab_path_list,
                                     location=patch_num)
                # (1, 128, 128, 128, 4) (128, 128, 128, 3)
                # print(one_case_fuse_data.shape, one_case_mask_data.shape)
                pre_mask = model.predict(one_case_fuse_data, batch_size=1)  # (1, 128, 128, 128, 3)
                predict_mask_01 = pre_mask > optimal_threshold
                data_sets.append(pre_mask)
                data_sets01.append(predict_mask_01)

            fused_predict_wt = fuse_array2complete_matrix(data_sets[0][0, :, :, :, 0], data_sets[1][0, :, :, :, 0],
                                                          data_sets[2][0, :, :, :, 0], data_sets[3][0, :, :, :, 0],
                                                          data_sets[4][0, :, :, :, 0], data_sets[5][0, :, :, :, 0],
                                                          data_sets[6][0, :, :, :, 0], data_sets[7][0, :, :, :, 0],
                                                          data_sets[8][0, :, :, :, 0])

            fused_predict_tc = fuse_array2complete_matrix(data_sets[0][0, :, :, :, 1], data_sets[1][0, :, :, :, 1],
                                                          data_sets[2][0, :, :, :, 1], data_sets[3][0, :, :, :, 1],
                                                          data_sets[4][0, :, :, :, 1], data_sets[5][0, :, :, :, 1],
                                                          data_sets[6][0, :, :, :, 1], data_sets[7][0, :, :, :, 1],
                                                          data_sets[8][0, :, :, :, 1])

            fused_predict_et = fuse_array2complete_matrix(data_sets[0][0, :, :, :, 2], data_sets[1][0, :, :, :, 2],
                                                          data_sets[2][0, :, :, :, 2], data_sets[3][0, :, :, :, 2],
                                                          data_sets[4][0, :, :, :, 2], data_sets[5][0, :, :, :, 2],
                                                          data_sets[6][0, :, :, :, 2], data_sets[7][0, :, :, :, 2],
                                                          data_sets[8][0, :, :, :, 2])
            fused_predict_wt_01 = (fused_predict_wt > 0.55).astype(np.int8)
            fused_predict_tc_01 = (fused_predict_tc > 0.50).astype(np.int8)
            fused_predict_et_01 = (fused_predict_et > 0.35).astype(np.int8)

            fused_predict = np.zeros_like(fused_predict_wt_01)
            fused_predict[fused_predict_wt_01 == 1] = 2
            fused_predict[fused_predict_tc_01 == 1] = 1
            fused_predict[fused_predict_et_01 == 1] = 4

            pre_filename_complete = create_case_path + ".nii.gz"      # multi-class label map  1 2 4
            pre_whole = create_case_path + "_unc_whole.nii.gz"          # 1 2 4
            pre_core = create_case_path + "_unc_core.nii.gz"            # 1 4
            pre_enhance = create_case_path + "_unc_enhance.nii.gz"      # 4

            print("filename_complete", pre_filename_complete)

            self.save_nib(fused_predict, pre_filename_complete)
            self.save_nib(fused_predict_wt_01, pre_whole)
            self.save_nib(fused_predict_tc_01, pre_core)
            self.save_nib(fused_predict_et_01, pre_enhance)
            # self.save_nib(fused_ground, true_filename_complete)
            """
            2. {ID}_unc_whole.nii.gz (Uncertainty map associated with whole tumor)
            3. {ID}_unc_core.nii.gz (Uncertainty map associated with tumor core)
            4. {ID}_unc_enhance.nii.gz (Uncertainty map associated with enhancing tumor)
            """


if __name__ == "__main__":
    # path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTS_2019_Validation_HM_Norm_719"
    # path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTs19Mixture_N4_HM_Norm/all"
    path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTS_2019_Validation_HM_Norm_719"
    # w_p = r"./checkpoint_813/model3d_818_01.h5"
    w_p = r"./checkpoint_813/model3d_825_01.h5"
    prediction = PredictCase()
    prediction.processing(w_p, path)


































