from keras.utils import Sequence
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import nibabel as nib
import os
from time import sleep
from patch_utils import get_patch_from_array_around_ranch


class CustomError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)   # 初始化父类
        self.errorinfo=ErrorInfo


class SequenceData(Sequence):
    def __init__(self, path, patch_size, batch_size):
        self.path = path
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.sub_dirs = glob.glob(path)
        self.data = []
        self.mask = []
        self.one_batch_data = None
        self.one_batch_mask = None

    def __len__(self):
        # steps
        num_case = len(self.sub_dirs)
        return math.ceil(num_case / self.batch_size)
        # return num_case

    def __getitem__(self, item):
        # 迭代器部分,一次迭代一个batch的数据
        self.read_data()
        self.one_batch_data = self.data[item * self.batch_size:(item + 1) * self.batch_size]
        self.one_batch_mask = self.mask[item * self.batch_size:(item + 1) * self.batch_size]
        return np.array(self.one_batch_data), np.array(self.one_batch_mask)

    def read_data(self):
        # 将所有的数据读到内存
        for index, sub_dir in enumerate(self.sub_dirs):
            data_dict = {}
            type_model = glob.glob(sub_dir + "/*")
            for sub_model in type_model:
                if sub_model.split(".")[0].split("_")[-1] == "flair":
                    data_dict["flair"] = self.get_path_from_array(nib.load(sub_model).get_data())
                elif sub_model.split(".")[0].split("_")[-1] == "t1":
                    data_dict["t1"] = self.get_path_from_array(nib.load(sub_model).get_data())
                elif sub_model.split(".")[0].split("_")[-1] == "t1ce":
                    data_dict["t1ce"] = self.get_path_from_array(nib.load(sub_model).get_data())
                elif sub_model.split(".")[0].split("_")[-1] == "t2":
                    data_dict["t2"] = self.get_path_from_array(nib.load(sub_model).get_data())
                elif sub_model.split(".")[0].split("_")[-1] == "seg":
                    mask = self.get_path_from_array(nib.load(sub_model).get_data())
            fuse_data = self.fuse_data(data_dict["flair"], data_dict["t1"], data_dict["t1ce"], data_dict["t2"])
            mask_data = self.label_bin_processing(mask)
            self.data.append(fuse_data)
            self.mask.append(mask_data)
        pass

    def get_path_from_array(self, image_array):
        """

        :param image_array:
        :return:
        """
        shape = image_array.shape
        patch_shape = self.patch_size
        Center_coordinate = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]
        assert (Center_coordinate[0] + int(patch_shape[0] / 2) <= shape[0]), "Out of size range"
        assert (Center_coordinate[0] - int(patch_shape[0] / 2) >= 0), "Out of size range"
        assert (Center_coordinate[1] + int(patch_shape[1] / 2) <= shape[0]), "Out of size range"
        assert (Center_coordinate[1] - int(patch_shape[1] / 2) >= 0), "Out of size range"
        assert (Center_coordinate[2] + int(patch_shape[2] / 2) <= shape[0]), "Out of size range"
        assert (Center_coordinate[2] - int(patch_shape[2] / 2) >= 0), "Out of size range"
        patch_data = \
            image_array[Center_coordinate[0] - int(patch_shape[0] / 2):Center_coordinate[0] + int(patch_shape[0] / 2),
            Center_coordinate[1] - int(patch_shape[1] / 2):Center_coordinate[1] + int(patch_shape[1] / 2),
            Center_coordinate[2] - int(patch_shape[2] / 2):Center_coordinate[2] + int(patch_shape[2] / 2)]
        return patch_data

    @staticmethod
    def fuse_data(data1, data2, data3, data4):
        y1 = data1[..., np.newaxis]
        y2 = data2[..., np.newaxis]
        y3 = data3[..., np.newaxis]
        y4 = data4[..., np.newaxis]
        fuse_data = np.concatenate((y1, y2, y3, y4), axis=-1)
        return fuse_data

    @staticmethod
    def label_bin_processing(label_data, region_type="whole", all_labels=True):
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
                seg_labels[:, :, :, idx] = (label_data == int(label_num[idx])).astype(np.int8)
            whole_mask = seg_labels[:, :, :, 0] + seg_labels[:, :, :, 1] + seg_labels[:, :, :, 2]
            whole_mask = (whole_mask > 0).astype(np.int16)
            fuse_mask.append(whole_mask)

            core_mask = seg_labels[:, :, :, 0] + seg_labels[:, :, :, 2]
            core_mask = (core_mask > 0).astype(np.int8)
            fuse_mask.append(core_mask)

            active_mask = seg_labels[:, :, :, 2]
            active_mask = (active_mask > 0).astype(np.int8)
            fuse_mask.append(active_mask)
        except Exception as error:  # 捕获所有可能发生的异常
            print("ERROR：", error)
        finally:
            pass
        if all_labels:
            fuse_mask = np.transpose(np.array(fuse_mask), [1, 2, 3, 0])
            # print("fuse_mask.shape", fuse_mask.shape)      # fuse_mask.shape (128, 128, 128, 3)
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

    def test(self):
        print(self.sub_dirs)


if __name__ == "__main__":
    r_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTs19Mixture_N4_HM_Norm/all/*"
    # r_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/test815_1/Test815/*"
    patch_shape = [160, 160, 128]
    a = SequenceData(r_path, patch_shape, 16)
    for i in range(3):
        b, c = a[i]
        print(b.shape, c.shape)
    pass



