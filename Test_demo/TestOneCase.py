import SimpleITK as sitk
import os
from keras.models import Model
import numpy as np
from Resnet50SmoothNet.Resnet50SmoothL2 import DfnSmoothNet as DfnSmoothNet1
from Resnet50SmoothNet.Resnet50SmoothL2_o import DfnSmoothNet_0 as DfnSmoothNet2
from Resnet50SmoothNet.vis_utils import show_img_multiplex_cutoff
from time import sleep
import matplotlib.pyplot as plt
import keras
import nibabel as nib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Save2Nii:
    def __init__(self, reference_path=None, lib_type="nib"):
        """

        :param reference_path: For the first time to define class Save2Nii,you should fill in the reference path.
        Reference data provides stored parameters.
        like this：
        F = Save2Nii(data_path)
        F(img, filename)
        """
        ref_path = \
            r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTs19Mixture_N4_HM_Norm/Train/BraTS19_2013_11_1/" \
            r"BraTS19_2013_11_1_flair.nii.gz"
        assert os.path.exists(ref_path), "The input reference data does not exist or is incorrect."
        if reference_path is None:
            self.reference_path = ref_path
        else:
            self.reference_path = reference_path
        self.lib_type = lib_type

    def __call__(self, array, save_name):
        """

        :param array:  3D array with shape (D, W, H)
        :param save_name:
        :return:
        """
        assert self.lib_type in ["nib", "sitk"]
        if self.lib_type == "nib":
            self.save2nii_nib(array, save_name)
        else:
            self.save2nii_sitk(array, save_name)

    def save2nii_sitk(self, array, save_name):
        """

        :param array:
        :param save_name:
        :return:
        """
        reference_image = sitk.ReadImage(self.reference_path)
        array = sitk.GetImageFromArray(array)
        array.SetSpacing(reference_image.GetSpacing())
        array.SetOrigin(reference_image.GetOrigin())
        sitk.WriteImage(array, save_name)

    def save2nii_nib(self, array, save_path):
        """

        :param array:
        :param save_path:
        :return:
        """
        _, affine, _ = ReadData.nii_data_read_nib(self.reference_path)
        new_data = nib.Nifti1Image(array, affine)
        nib.save(new_data, save_path)
        # print("*******************")
        # print("nii done")
        # print("*******************")
        return new_data


class ReadData:
    def __init__(self, data_path,  default_lib="nib"):
        """

        :param data_path: nii data path
        """
        assert default_lib in ["nib", "sitk"], "Support only nibabel and SimpleITK, default is nibabel," \
                                               "The value of the default keyword must be 'nib' or 'sitk'. "
        self.data_path = data_path
        self.default_lib = default_lib
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __le__(self, other):
        pass

    def read_nii_data(self):
        if self.default_lib == "nib":
            case_data, _, _ = self.nii_data_read_nib(self.data_path)
        else:
            case_data = self.nii_data_read_sitk(self.data_path)

        return case_data

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

    @staticmethod
    def nii_data_read_sitk(data_path):
        """

        :return:
        """
        img = sitk.ReadImage(data_path)
        img_array = sitk.GetArrayFromImage(img)
        return img_array


class TestOneCase:
    def __init__(self, root_path, weight_wt, weight_tc, weight_et):
        """

        :param root_path:
        :param weight_wt:
        :param weiget_tc:
        :param weight_et:
        """
        self.root_path = root_path
        self.weight_wt = weight_wt
        self.weight_tc = weight_tc
        self.weight_et = weight_et
        self.nii_model_path = {}
        self.no_label_mask = np.zeros([240, 240])
        self.one_case_wt_mask = []
        self.one_case_tc_mask = []
        self.one_case_et_mask = []
        self.seg_data = 0

        # self.wt_model = self.base_model_1()
        # self.tc_model = self.base_model_2()
        # self.et_model = self.base_model_1()
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __le__(self, other):
        pass

    def processing(self, case_name, save_path):
        """

        :param case_name: One case name
        :return: mask.nii.gz (D, H, W)  sitk
        """
        # one_case_wt_mask = []
        # one_case_tc_mask = []
        # one_case_et_mask = []
        # nii_model_path = {}
        ab_case_path = os.path.join(self.root_path, case_name)
        model_nii_list = os.listdir(ab_case_path)
        global flair_data, t1_data, t1ce_data, t2_data, fuse_data
        for index, model_nii in enumerate(model_nii_list):
            # print(model_nii)      # t1ce, flair, t2, t1
            if model_nii.split(".")[0].split("_")[-1] == "t1ce":
                self.nii_model_path["t1ce"] = os.path.join(ab_case_path, model_nii)
                t1ce_data = ReadData(self.nii_model_path["t1ce"]).read_nii_data()
                print(" self.nii_model_path[t1ce]",  self.nii_model_path["t1ce"])
                # print(t1ce_data.shape)

            elif model_nii.split(".")[0].split("_")[-1] == "flair":
                self.nii_model_path["flair"] = os.path.join(ab_case_path, model_nii)
                flair_data = ReadData(self.nii_model_path["flair"]).read_nii_data()

            elif model_nii.split(".")[0].split("_")[-1] == "t2":
                self.nii_model_path["t2"] = os.path.join(ab_case_path, model_nii)
                t2_data = ReadData(self.nii_model_path["t2"]).read_nii_data()

            elif model_nii.split(".")[0].split("_")[-1] == "t1":
                self.nii_model_path["t1"] = os.path.join(ab_case_path, model_nii)
                t1_data = ReadData(self.nii_model_path["t1"]).read_nii_data()
            else:
                self.nii_model_path["seg"] = os.path.join(ab_case_path, model_nii)
                self.seg_data = ReadData(self.nii_model_path["seg"]).read_nii_data()

        # self.sub_process(flair_data, t1_data, t1ce_data, t2_data, model_type="wt")
        one_case_wt_mask = self.sub_process(flair_data, t1_data, t1ce_data, t2_data, model_type="wt")
        one_case_wt_mask_copy = self.sub_process(flair_data, t1_data, t1ce_data, t2_data, model_type="wt")
        one_case_tc_mask = self.sub_process(flair_data, t1_data, t1ce_data, t2_data, model_type="tc")
        one_case_et_mask = self.sub_process(flair_data, t1_data, t1ce_data, t2_data, model_type="et")
        # one_case_wt_mask_copy = one_case_wt_mask.copy
        one_case_data = self.fuse_data(one_case_wt_mask, one_case_tc_mask, one_case_et_mask)

        one_case_data_name = case_name + ".nii.gz"
        whole_tumor = case_name + "_unc_whole.nii.gz"
        tumor_core = case_name + "_unc_core.nii.gz"
        enhancing_tumor = case_name + "_unc_enhance.nii.gz"
        ab_path = os.path.join(save_path, case_name)

        if not os.path.exists(ab_path):
            os.makedirs(ab_path)
        final_ab_whole_tumor_path = os.path.join(ab_path, whole_tumor)
        final_ab_tumor_core_path = os.path.join(ab_path, tumor_core)
        final_ab_enhancing_tumor_path = os.path.join(ab_path, enhancing_tumor)
        final_ab_path = os.path.join(ab_path, one_case_data_name)

        print("******", final_ab_path)
        save2nii = Save2Nii()

        save2nii.save2nii_nib(one_case_data, final_ab_path)

        save2nii.save2nii_nib(one_case_wt_mask_copy, final_ab_whole_tumor_path)
        save2nii.save2nii_nib(one_case_tc_mask, final_ab_tumor_core_path)
        save2nii.save2nii_nib(one_case_et_mask, final_ab_enhancing_tumor_path)

        print("{} has been processed !".format(case_name))

        # print("one_case_wt_mask", one_case_wt_mask.shape)     # (240, 240, 155)
        # print("one_case_tc_mask", one_case_tc_mask.shape)
        # print("one_case_et_mask", one_case_et_mask.shape)

    @staticmethod
    def fuse_data(wt_matrix, tc_matrix, et_matrix):
        """
        wt: 1, 2, 4   tc: 1, 4   et: 4
        :param wt_matrix:
        :param tc_matrix:
        :param et_matrix:
        :return:
        """
        assert wt_matrix.shape == tc_matrix.shape == et_matrix.shape, "Please check the input data"
        temporary_matrix = wt_matrix
        temporary_matrix[wt_matrix == 1] = 2
        temporary_matrix[tc_matrix == 1] = 1
        temporary_matrix[et_matrix == 1] = 4
        return temporary_matrix

    def sub_process(self, flair_array, t1_array, t1ce_array, t2_array, model_type):
        """

        :param flair_array:
        :param t1_array:
        :param t1ce_array:
        :param t2_array:
        :param model_type:
        :return:
        """
        assert model_type in ["wt", "et", "tc"]
        one_case_mask = []
        shape = flair_array.shape
        keras.backend.clear_session()

        if model_type == "wt":
            current_model = self.base_model_1()
            current_model.load_weights(self.weight_wt)
        elif model_type == "tc":
            current_model = self.base_model_2()
            current_model.load_weights(self.weight_tc)
        else:
            current_model = self.base_model_1()
            current_model.load_weights(self.weight_et)
        print("Initializing parameter weights ...")
        for idx_1 in range(shape[2]):
            if idx_1 == 0:
                flag = "top_slices"
            elif idx_1+1 == shape[2]:
                flag = "bottom_slices"
            else:
                flag = "middle_slices"
            slice_data = flair_array[:, :, idx_1]
            if slice_data.sum() == 0:
                one_case_mask.append(self.no_label_mask)
            else:
                flair_three_slices = self.read_three_layers_onetime(flair_array, idx_1, flag=flag)
                # print(flair_three_slices)
                t1_three_slices = self.read_three_layers_onetime(t1_array, idx_1, flag=flag)
                t1ce_three_slices = self.read_three_layers_onetime(t1ce_array, idx_1, flag=flag)
                t2_three_slices = self.read_three_layers_onetime(t2_array, idx_1, flag=flag)
                fuse_array = \
                    np.concatenate((flair_three_slices, t1_three_slices, t1ce_three_slices, t2_three_slices), axis=0)
                fuse_array = np.transpose(fuse_array, [1, 2, 0])
                fuse_array = fuse_array[np.newaxis, ...]      # (1, 240, 240, 12)
                slice_mask = current_model.predict(fuse_array, batch_size=1)[0]    # (1, 240, 240, 1)
                # print("slice_mask", slice_mask.shape)
                if model_type == "wt":
                    one_case_mask.append((slice_mask[0, :, :, 0] > 0.5).astype(int))
                elif model_type == "tc":
                    one_case_mask.append((slice_mask[0, :, :, 0] > 0.5).astype(int))
                else:
                    one_case_mask.append((slice_mask[0, :, :, 0] > 0.36).astype(int))

                # seg_slice_data = self.seg_data[:, :, idx_1]

                # # 可视化
                # slice_name = case_name + "_" + str(idx_1)
                # show_img_multiplex_cutoff(slice_mask[0, :, :, 0], seg_slice_data, cutoff=0.5, img_title=slice_name)
                #
                # sleep(1)

        one_case_mask = np.transpose(np.array(one_case_mask), [1, 2, 0])
        print("{} has done".format(model_type))
        return one_case_mask

    @staticmethod
    def show_img_multiplex(predict_mask, true_mask, img_title="image"):
        plt.figure()
        plt.title(img_title)  # 图像题目
        plt.subplot(1, 2, 1)
        plt.title("sitk_data")
        plt.imshow(predict_mask, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("nib_data")
        plt.imshow(true_mask, cmap='gray')
        plt.suptitle(img_title)
        plt.show()

    @staticmethod
    def base_model_1(load_weight=None):
        train_model = DfnSmoothNet1()
        "model output: [b0, b1, b2, b3, b4, fuse]"
        model = Model(inputs=[train_model.input], outputs=[train_model.output[0],
                                                           train_model.output[1],
                                                           train_model.output[2],
                                                           train_model.output[3],
                                                           train_model.output[4],
                                                           train_model.output[5]])
        if load_weight:
            model.load_weights(load_weight)
        return model

    @staticmethod
    def base_model_2(load_weight=None):
        train_model = DfnSmoothNet2()
        "model output: [b0, b1, b2, b3, b4, fuse]"
        model = Model(inputs=[train_model.input], outputs=[train_model.output[0],
                                                           train_model.output[1],
                                                           train_model.output[2],
                                                           train_model.output[3],
                                                           train_model.output[4],
                                                           train_model.output[5]])
        if load_weight:
            model.load_weights(load_weight)
        return model

    @staticmethod
    def read_three_layers_onetime(array_data, idx, flag="middle_slices", data_type="nib"):
        """

        :param array_data:
        :param idx:
        :param flag:
        :param data_type:
        :return:
        """

        assert flag in ["middle_slices", "top_slices", "bottom_slices"]
        assert data_type in ["nib", "sitk"]
        three_slice = []
        if data_type == "sitk":
            if flag == "top_slices":
                three_slice.append(array_data[idx, :, :])
                three_slice.append(array_data[idx+1, :, :])
                three_slice.append(array_data[idx+2, :, :])

            elif flag == "bottom_slices":
                three_slice.append(array_data[idx, :, :])
                three_slice.append(array_data[idx-1, :, :])
                three_slice.append(array_data[idx-2, :, :])

            else:
                three_slice.append(array_data[idx-1, :, :])
                three_slice.append(array_data[idx, :, :])
                three_slice.append(array_data[idx+1, :, :])
        else:
            if flag == "top_slices":
                three_slice.append(array_data[:, :, idx])
                three_slice.append(array_data[:, :, idx + 1])
                three_slice.append(array_data[:, :, idx + 2])

            elif flag == "bottom_slices":
                three_slice.append(array_data[:, :, idx])
                three_slice.append(array_data[:, :, idx - 1])
                three_slice.append(array_data[:, :, idx - 2])

            else:
                three_slice.append(array_data[:, :, idx - 1])
                three_slice.append(array_data[:, :, idx])
                three_slice.append(array_data[:, :, idx + 1])

        return np.asarray(three_slice)


def base_model_1(load_weight=None):
    train_model = DfnSmoothNet1()
    "model output: [b0, b1, b2, b3, b4, fuse]"
    model = Model(inputs=[train_model.input], outputs=[train_model.output[0],
                                                       train_model.output[1],
                                                       train_model.output[2],
                                                       train_model.output[3],
                                                       train_model.output[4],
                                                       train_model.output[5]])
    if load_weight:
        pass
        model.load_weights(load_weight)
    return model


def base_model_2(load_weight=None):
    train_model = DfnSmoothNet2()
    "model output: [b0, b1, b2, b3, b4, fuse]"
    model = Model(inputs=[train_model.input], outputs=[train_model.output[0],
                                                       train_model.output[1],
                                                       train_model.output[2],
                                                       train_model.output[3],
                                                       train_model.output[4],
                                                       train_model.output[5]])
    if load_weight:
        pass
        model.load_weights(load_weight)
    return model


def test_ensemble_model(model1_path, model2_path):
    model1 = base_model_1(model1_path)
    # model1.load_weights(model1_path)

    keras.backend.clear_session()

    model2 = base_model_2(model2_path)
    # model2.load_weights(model2_path)
    pass


if __name__ == "__main__":
    path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTS_2019_Validation_HM_Norm_719"
    seg_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/Resust_mask729"
    w_root_path = r"/home/yk/Project/keras/DFNSmoothNet/Resnet50SmoothNet/checkpoint_new2"
    w_wt = r"dfn_norm_wt_630_002.h5"
    w_tc = r"dfn_norm_tc_0716_001.h5"
    w_et = r"dfn_norm_et_0719_001.h5"
    w_wt_path = os.path.join(w_root_path, w_wt)
    w_tc_path = os.path.join(w_root_path, w_tc)
    w_et_path = os.path.join(w_root_path, w_et)

    F = TestOneCase(path, w_wt_path, w_tc_path, w_et_path)
    case_name_list = os.listdir(path)
    all_case_num = len(case_name_list)
    for idx, case_name in enumerate(case_name_list):
        F.processing(case_name, seg_path)
        print("{}/{}  {} has been processed !".format(idx+1, all_case_num, case_name))

    # test_ensemble_model(w_wt_path, w_tc_path)
    # pass






































