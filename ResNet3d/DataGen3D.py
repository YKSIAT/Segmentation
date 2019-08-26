import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import glob


class CustomError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)   # 初始化父类
        self.errorinfo=ErrorInfo


def GetPathFromArray(image_array, p_shape):
    """

    :param image_array:
    :param p_shape:
    :return:
    """
    # residual_top = np.zeros((image_array.shape[0], image_array.shape[1], 2))
    # residual_bottom = np.zeros((image_array.shape[0], image_array.shape[1], 3))
    # image_array = np.concatenate((residual_top, image_array, residual_bottom),  axis=-1)
    # if p_shape[2] > image_array.shape[2]:
    #     residual_num = p_shape[2] - image_array.shape[2]
    #     residual_bottom = np.zeros((image_array.shape[0], image_array.shape[1], residual_num))

    shape = image_array.shape
    # print("image_array.shape", image_array.shape)
    patch_shape = p_shape
    # Center_coordinate = [120, 120, 77]
    Center_coordinate = [int(shape[0]/2), int(shape[1]/2), int(shape[2]/2)]
    # print(Center_coordinate[0] - int(patch_shape[0] / 2))
    # print(Center_coordinate[0] + int(patch_shape[0] / 2))
    # print(shape)
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


# def GetPathFromArray(image_array, p_shape):
#     """
#
#     :param image_array:
#     :param p_shape:
#     :return:
#     """
#     # print(image_array.shape)
#     # print(p_shape)
#     # residual_top = np.zeros((image_array.shape[0], image_array.shape[1], 2))
#     # residual_bottom = np.zeros((image_array.shape[0], image_array.shape[1], 3))
#     # image_array = np.concatenate((residual_top, image_array, residual_bottom),  axis=-1)
#
#     if p_shape[2] > image_array.shape[2]:
#         residual_num = p_shape[2] - image_array.shape[2]
#         residual_bottom = np.zeros((image_array.shape[0], image_array.shape[1], residual_num))
#     image_array = np.concatenate([image_array, residual_bottom], axis=-1)
#     shape = image_array.shape
#     # print("image_array.shape", image_array.shape)
#     patch_shape = p_shape
#     # Center_coordinate = [120, 120, 77]
#     Center_coordinate = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]
#
#     assert (Center_coordinate[0] + int(patch_shape[0] / 2) <= shape[0]), "Out of size range"
#     assert (Center_coordinate[0] - int(patch_shape[0] / 2) >= 0), "Out of size range"
#
#     assert (Center_coordinate[0] + int(patch_shape[1] / 2) <= shape[0]), "Out of size range"
#     assert (Center_coordinate[0] - int(patch_shape[1] / 2) >= 0), "Out of size range"
#
#     assert (Center_coordinate[0] + int(patch_shape[2] / 2) <= shape[0]), "Out of size range"
#     assert (Center_coordinate[0] - int(patch_shape[2] / 2) >= 0), "Out of size range"
#
#     patch_data = image_array[
#                  Center_coordinate[0] - int(patch_shape[0] / 2):Center_coordinate[0] + int(patch_shape[0] / 2),
#                  Center_coordinate[1] - int(patch_shape[1] / 2):Center_coordinate[1] + int(patch_shape[1] / 2),
#                  Center_coordinate[2] - int(patch_shape[2] / 2):Center_coordinate[2] + int(patch_shape[2] / 2)]
#     # print("patch_data", patch_data.shape)
#     return patch_data


def FuseData(data1, data2, data3, data4):
    y1 = data1[..., np.newaxis]
    y2 = data2[..., np.newaxis]
    y3 = data3[..., np.newaxis]
    y4 = data4[..., np.newaxis]
    fuse_data = np.concatenate((y1, y2, y3, y4), axis=-1)
    return fuse_data


# def label_bin_processing(label_data):
#
#     label_num = [1, 2, 4]
#     label_data_shape = label_data.shape
#     assert len(label_data_shape) == 3, "The shape of label data should be 3d"
#     seg_labels = np.zeros((label_data_shape[0], label_data_shape[1], label_data_shape[2], len(label_num)))
#     # label_data[:, :, :][label_data[:, :, :] == 4] = 3
#     for idx in range(len(label_data_shape)):
#         seg_labels[:, :, :, idx] = (label_data == int(label_num[idx])).astype(int)
#     return seg_labels


def label_bin_processing(label_data,  region_type="whole", all_labels=True):
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


def Gen3D(root_path, patch_shape, batch_size=2):
    sub_dirs = glob.glob(root_path)
    print(len(sub_dirs))
    # print("sub_dirs", sub_dirs)
    while 1:
        case_list = []
        image_data_batch = []
        mask_data_batch = []
        # for idx1, sub_dir in enumerate(sub_dirs):
        #     case_list.extend(glob.glob(sub_dir + "/*"))
        # print("The number of case is {}".format(len(case_list)))
        for idx, dir in enumerate(sub_dirs):
            data = []
            data_dict = {}
            type_model = glob.glob(dir + "/*")
            # print("type_model", type_model)
            for sub_model in type_model:
                # print(sub_model)
                # print(sub_model.split(".")[0].split("_")[-1])
                if sub_model.split(".")[0].split("_")[-1] == "flair":
                    data_dict["flair"] = GetPathFromArray(nib.load(sub_model).get_data(), patch_shape)
                elif sub_model.split(".")[0].split("_")[-1] == "t1":
                    data_dict["t1"] = GetPathFromArray(nib.load(sub_model).get_data(), patch_shape)
                elif sub_model.split(".")[0].split("_")[-1] == "t1ce":
                    data_dict["t1ce"] = GetPathFromArray(nib.load(sub_model).get_data(), patch_shape)
                elif sub_model.split(".")[0].split("_")[-1] == "t2":
                    data_dict["t2"] = GetPathFromArray(nib.load(sub_model).get_data(), patch_shape)
                # if sub_model.split(".")[0].split("_")[-1] != "seg":
                #     # print("1")
                #     data.append(GetPathFromArray(nib.load(sub_model).get_data(), patch_shape))

                elif sub_model.split(".")[0].split("_")[-1] == "seg":
                    # print("2")
                    mask = GetPathFromArray(nib.load(sub_model).get_data(), patch_shape)
            # fuse_data = FuseData(data[0], data[1], data[2], data[3])
            fuse_data = FuseData(data_dict["flair"], data_dict["t1"], data_dict["t1ce"], data_dict["t2"])
            mask_data = label_bin_processing(mask)
            image_data_batch.append(fuse_data)
            mask_data_batch.append(mask_data)
            if len(image_data_batch) % batch_size == 0 or idx + 1 == len(case_list):
                yield np.array(image_data_batch), np.array(mask_data_batch)
                # print("hello")
                image_data_batch.clear()
                mask_data_batch.clear()


def show_img(predict_mask, true_mask, img_title="image"):
    plt.figure()
    plt.title(img_title)  # 图像题目
    plt.subplot(1, 2, 1)
    plt.title("predict_mask")
    plt.imshow(predict_mask, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("true_mask")
    plt.imshow(true_mask, cmap='gray')
    plt.suptitle(img_title)
    plt.show()


if __name__ == "__main__":
    r_path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/BraTs19Mixture_N4_HM_Norm/all/*"
    patch_shape = [160, 160, 128]
    gen = Gen3D(r_path, patch_shape=patch_shape)
    for idx, value in enumerate(gen):
        print(value[0].shape)
        print(value[1].shape)



