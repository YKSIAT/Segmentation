import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from glob import glob
from skimage import transform
# from keras.layers import *
import shutil
import warnings

warnings.filterwarnings('ignore')


class CustomError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)     # 初始化父类
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo


def read_img(data_path, header=False):
    """

    :param : data_path
    :param : with header or not
    :return: (240, 240, 155) <class 'numpy.ndarray'>
    """
    img = nib.load(data_path)
    image_array_data = img.get_data()
    # image_array_data = np.transpose(image_array_data, [2, 1, 0])   # (155, 240, 240)
    if header:
        return image_array_data, img.affine, img.header
    else:
        return image_array_data
    # return image_array_data


# def read_img(path):
#     """
#
#     :param path:
#     :return: (155, 240, 240)  <class 'numpy.ndarray'>
#     """
#     img = sitk.ReadImage(path)    # <class 'SimpleITK.SimpleITK.Image'>
#     data = sitk.GetArrayFromImage(img)
#     return data   # (z, x, y)

def save_array_as_nifty_volume(data, filename, reference_name=None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if reference_name is not None:
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


def show_img(ori_img, title="image"):
    plt.title = title
    plt.imshow(ori_img[0, :, :], cmap='gray')  # channel_last(x, y, z)
    plt.show()


def show_img_multiplex(mask, whole, core, active, img_title="image"):
    plt.figure()
    plt.title(img_title)  # 图像题目
    plt.subplot(2, 2, 1)
    plt.title("mask")
    mask = mask[0, :, :]
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title("whole")
    whole = whole[0, :, :]
    plt.imshow(whole, cmap='gray')
    plt.suptitle(img_title)
    plt.subplot(2, 2, 3)
    plt.title("core")
    core = core[0, :, :]
    plt.imshow(core, cmap='gray')
    plt.suptitle(img_title)
    plt.subplot(2, 2, 4)
    plt.title("active")
    active = active[0, :, :]
    plt.imshow(active, cmap='gray')
    plt.suptitle(img_title)
    plt.show()


def get_mask_array(mask_slice_data, regions_type="whole", all_labels=False):
    """

    :param mask_slice_data:(1, 240, 240)
    :param regions_type   This parameter determines the type of label returned (whole, core or active)
    :param all_labels     This parameter determines if return all type including whole, core or active
    if it's True return all three regions ,otherwise ,return one of regions
    :return: 0, 1 n_dim
    """
    regions_types = ["whole", "core", "active"]
    assert regions_type in regions_types, "The parameter value of regions_type is wrong, " \
                                          "and it needs to be selected from 'whole', 'core', 'active' "
    # mask_slice_data = mask_slice_data[0, :, :]
    label_num = [1, 2, 4]
    mask_slice_data_shape = mask_slice_data.shape
    seg_labels = np.zeros((mask_slice_data_shape[0], mask_slice_data_shape[1], len(label_num)))
    # three_stacked = np.zeros((mask_slice_data_shape[0], mask_slice_data_shape[1], len(label_num)))
    whole_mask = np.zeros((mask_slice_data_shape[0], mask_slice_data_shape[1]), dtype=np.uint8)
    core_mask = np.zeros((mask_slice_data_shape[0], mask_slice_data_shape[1]), dtype=np.uint8)
    active_mask = np.zeros((mask_slice_data_shape[0], mask_slice_data_shape[1]), dtype=np.uint8)

    try:

        # for c in range(n_classes):
        for idx in range(len(label_num)):
            # show_img(mask_slice_data)
            seg_labels[:, :, idx] = (mask_slice_data == int(label_num[idx])).astype(int)
        whole_mask = seg_labels[:, :, 0] + seg_labels[:, :, 1] + seg_labels[:, :, 2]
        core_mask = seg_labels[:, :, 0] + seg_labels[:, :, 2]
        active_mask = seg_labels[:, :, 2]

    except Exception as error:  # 捕获所有可能发生的异常
        print("ERROR：", error)
    finally:
        pass
        # print("Done！")
    three_stacked = np.dstack((whole_mask, core_mask, active_mask))
    if all_labels:
        return three_stacked

    else:
        if regions_type == "whole":
            return whole_mask
        elif regions_type == "core":
            return core_mask
        elif regions_type == "active":
            return active_mask
        else:
            raise CustomError('Parameter values need to be selected from "whole", "core" and "active"')
    # seg_labels = np.reshape(seg_labels, (width * height, n_classes))
    # return whole_mask


def standardization(model_data):
    mean = np.mean(model_data)
    std = np.std(model_data)
    result = (model_data - mean)/std
    return result


def get_near_the_three_layers(root_path, save_path):
    sub_dirs = [x[0] for x in os.walk(root_path)]
    is_root_dir = True
    model_name = ["flair", "t1", "t1ce", "t2", "seg"]
    # print(sub_dirs)
    for index, sub_dir in enumerate(sub_dirs):
        # print(index, sub_dir)
        if is_root_dir:
            is_root_dir = False
            continue
        # print(index, sub_dir)
        # print(sub_dir, os.path.basename(sub_dir))     # H:\2018\N4\Brats18_TCIA13_645_1 Brats18_TCIA13_645_1
        files_list = os.listdir(sub_dir)   # get all folders
        # print(files_list)
        model_data = dict()
        image_save_path = os.path.join(save_path, os.path.basename(os.path.normpath(sub_dir)), "fusion_images")
        mask_save_path = os.path.join(save_path, os.path.basename(os.path.normpath(sub_dir)), "masks")
        whole_mask_save_path = os.path.join(save_path, os.path.basename(os.path.normpath(sub_dir)), "whole_wt")
        core_mask_save_path = os.path.join(save_path, os.path.basename(os.path.normpath(sub_dir)), "core_tc")
        active_mask_save_path = os.path.join(save_path, os.path.basename(os.path.normpath(sub_dir)), "active_et")

        # print("image_save_path", image_save_path)
        # print("image_save_path", mask_save_path)
        # print()
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)
        if not os.path.exists(whole_mask_save_path):
            os.makedirs(whole_mask_save_path)
        if not os.path.exists(core_mask_save_path):
            os.makedirs(core_mask_save_path)
        if not os.path.exists(active_mask_save_path):
            os.makedirs(active_mask_save_path)
        for item in files_list:
            image_model = item.split(".")[0].strip().split("_")[-1].strip()
            if image_model == model_name[0]:
                model_data[model_name[0]] = os.path.join(sub_dir, item)
            elif image_model == model_name[1]:
                model_data[model_name[1]] = os.path.join(sub_dir, item)
            elif image_model == model_name[2]:
                model_data[model_name[2]] = os.path.join(sub_dir, item)
            elif image_model == model_name[3]:
                model_data[model_name[3]] = os.path.join(sub_dir, item)
            else:
                model_data[model_name[4]] = os.path.join(sub_dir, item)
                # print(model_name[4], model_data[model_name[4]])
        mask_data = read_img(model_data[model_name[4]])
        flair_data = read_img(model_data[model_name[0]])
        t1_data = read_img(model_data[model_name[1]])
        t1ce_data = read_img(model_data[model_name[2]])
        t2_data = read_img(model_data[model_name[3]])
        shape = mask_data.shape    # nib (240, 240, 155)
        counter = 0
        for idx in range(shape[2]):
            # mask_slice_data_whole = []
            # mask_slice_data_core = []
            # mask_slice_data_active = []
            # mask_slice_data_ori = []

            # mask_slice_data_ = mask_data[:, :, idx]    # sitk
            mask_slice_data = mask_data[:, :, idx]
            if mask_slice_data.sum() != 0:
                get_mask_array(mask_slice_data, regions_type="whole", all_labels=False)
                mask_slice_data_ori = mask_slice_data
                mask_slice_data_whole = (get_mask_array(mask_slice_data, regions_type="whole"))
                mask_slice_data_core = (get_mask_array(mask_slice_data, regions_type="core"))
                mask_slice_data_active = (get_mask_array(mask_slice_data, regions_type="active"))

                flair_slice_data = []
                t1_slice_data = []
                t1ce_slice_data = []
                t2_slice_data = []
                if idx == 0:
                    flair_slice_data.append(flair_data[:, :, idx])
                    flair_slice_data.append(flair_data[:, :, idx])
                    flair_slice_data.append(flair_data[:, :, idx+1])
                    flair_slice_data = np.array(flair_slice_data)

                    t1_slice_data.append(t1_data[:, :, idx])
                    t1_slice_data.append(t1_data[:, :, idx])
                    t1_slice_data.append(t1_data[:, :, idx+1])
                    t1_slice_data = np.array(t1_slice_data)

                    t1ce_slice_data.append(t1ce_data[:, :, idx])
                    t1ce_slice_data.append(t1ce_data[:, :, idx])
                    t1ce_slice_data.append(t1ce_data[:, :, idx+1])
                    t1ce_slice_data = np.array(t1ce_slice_data)

                    t2_slice_data.append(t2_data[:, :, idx])
                    t2_slice_data.append(t2_data[:, :, idx])
                    t2_slice_data.append(t2_data[:, :, idx+1])
                    t2_slice_data = np.array(t2_slice_data)

                elif idx == shape[2]-1:
                    flair_slice_data.append(flair_data[:, :, idx-1])
                    flair_slice_data.append(flair_data[:, :, idx])
                    flair_slice_data.append(flair_data[:, :, idx])
                    flair_slice_data = np.array(flair_slice_data)

                    t1_slice_data.append(t1_data[:, :, idx-1])
                    t1_slice_data.append(t1_data[:, :, idx])
                    t1_slice_data.append(t1_data[:, :, idx])
                    t1_slice_data = np.array(t1_slice_data)

                    t1ce_slice_data.append(t1ce_data[:, :, idx-1])
                    t1ce_slice_data.append(t1ce_data[:, :, idx])
                    t1ce_slice_data.append(t1ce_data[:, :, idx])
                    t1ce_slice_data = np.array(t1ce_slice_data)

                    t2_slice_data.append(t2_data[:, :, idx-1])
                    t2_slice_data.append(t2_data[:, :, idx])
                    t2_slice_data.append(t2_data[:, :, idx])
                    t2_slice_data = np.array(t2_slice_data)
                else:
                    flair_slice_data.append(flair_data[:, :, idx-1])
                    flair_slice_data.append(flair_data[:, :, idx])
                    flair_slice_data.append(flair_data[:, :, idx+1])
                    flair_slice_data = np.array(flair_slice_data)

                    t1_slice_data.append(t1_data[:, :, idx-1])
                    t1_slice_data.append(t1_data[:, :, idx])
                    t1_slice_data.append(t1_data[:, :, idx+1])
                    t1_slice_data = np.array(t1_slice_data)

                    t1ce_slice_data.append(t1ce_data[:, :, idx-1])
                    t1ce_slice_data.append(t1ce_data[:, :, idx])
                    t1ce_slice_data.append(t1ce_data[:, :, idx+1])
                    t1ce_slice_data = np.array(t1ce_slice_data)

                    t2_slice_data.append(t2_data[:, :, idx-1])
                    t2_slice_data.append(t2_data[:, :, idx])
                    t2_slice_data.append(t2_data[:, :, idx+1])
                    t2_slice_data = np.array(t2_slice_data)
                    # print("t2_slice_data", t2_slice_data.shape)    # (3, 240, 240)
                data_fusion = np.concatenate((flair_slice_data, t1_slice_data, t1ce_slice_data, t2_slice_data), axis=0)
                # print("data_fusion.shape", data_fusion.shape)     # (12, 240, 240)

                print("mask_slice_data_ori", mask_slice_data_ori.shape)          # (240, 240)
                print("mask_slice_data_whole", mask_slice_data_whole.shape)      # (240, 240)
                print("mask_slice_data_core", mask_slice_data_core.shape)        # (240, 240)
                print("mask_slice_data_active", mask_slice_data_active.shape)    # (240, 240)

                # print("mask_slice_data.shape", mask_slice_data.shape)    # (1, 240, 240)
                counter += 1

                slice_name = os.path.basename(os.path.normpath(sub_dir)) + "_" + str(idx)
                image_slice_save_path = os.path.join(image_save_path, slice_name)
                mask_slice_save_path = os.path.join(mask_save_path, slice_name)
                whole_mask_slice_save_path = os.path.join(whole_mask_save_path, slice_name)
                core_mask_slice_save_path = os.path.join(core_mask_save_path, slice_name)
                active_mask_slice_save_path = os.path.join(active_mask_save_path, slice_name)

                np.save(image_slice_save_path, data_fusion)
                np.save(mask_slice_save_path, mask_slice_data_ori)
                np.save(whole_mask_slice_save_path, mask_slice_data_whole)
                np.save(core_mask_slice_save_path, mask_slice_data_core)
                np.save(active_mask_slice_save_path, mask_slice_data_active)
        # print("sub_dir", sub_dir)
        print("{} has {} slices with tumor labels ".format(sub_dir.split("\\")[-1], counter))
        print()

    pass


def generator_image_mask_three(data_path, batch_size=8):
    """

    :param data_path: data_path/patients/fusion_images/*.npy  data_path/patients/masks/*.npy
    :param batch_size:
    :return:
    """
    while True:
        all_data_slices = glob(os.path.join(data_path, "*", "fusion_images", "*.npy"))
        # all_mask_slices = glob(os.path.join(data_path, "*", "masks", "*.npy"))
        # all_mask_slices = glob(os.path.join(data_path, "*", "active_et", "*.npy"))
        # all_mask_slices = glob(os.path.join(data_path, "*", "core_tc", "*.npy"))
        all_mask_slices = glob(os.path.join(data_path, "*", "whole_wt", "*.npy"))
        assert len(all_data_slices) == len(all_mask_slices), \
            "The number of pictures and the number of labels must be the same"
        # print(len(all_data_slices))
        # print(len(all_mask_slices))
        image_batch_data = []
        mask_batch_data = []
        for idx, value in enumerate(zip(all_data_slices, all_mask_slices)):
            # print(np.load(value[1]).shape)

            image_batch_data.append(np.load(value[0]))
            mask_batch_data.append([np.load(value[1])])
            if idx == 0:
                continue
            if (idx + 1) % batch_size == 0:
                image_batch_data_np = np.transpose(np.array(image_batch_data), [0, 2, 3, 1])
                mask_batch_data_np = np.transpose(np.array(mask_batch_data), [0, 2, 3, 1])
                # print(image_batch_data_np.shape)
                yield (image_batch_data_np, mask_batch_data_np)
                image_batch_data.clear()
                mask_batch_data.clear()
            if idx + 1 == len(all_data_slices):
                image_batch_data_np = np.transpose(np.array(image_batch_data), [0, 2, 3, 1])   # (n, 240, 240, 1)
                mask_batch_data_np = np.transpose(np.array(mask_batch_data), [0, 2, 3, 1])     # (n, 240, 240, 12)
                yield (image_batch_data_np, mask_batch_data_np)
                # print(image_batch_data_np.shape)
                image_batch_data.clear()
                mask_batch_data.clear()

def generator_image_mask_three_test(data_path, batch_size=8):
    """
    From one file
    :param data_path:   data_path/whole_wt/*.npy
    :param batch_size:
    :return:
    """
    all_data_slices = glob(os.path.join(data_path, "whole_wt", "*.npy"))
    print(len(all_data_slices))
    files = os.listdir(data_path)
    for idx, value in enumerate(files):
        print(idx, value)


if __name__ == "__main__":
    mask_list = []
    p = [r"H:\N42HM\HistogramMatching\Train", r"H:\N42HM\HistogramMatching\Val"]
    s = [r"H:\N42HM\DataNpy_HM\Train", r"H:\N42HM\DataNpy_HM\Val"]
    for idx, path_ in enumerate(p):
        get_near_the_three_layers(p[idx], s[idx])

