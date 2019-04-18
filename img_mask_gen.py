import nibabel as nib
import numpy as np
import os
import random
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import io, transform
import math
import warnings

warnings.filterwarnings('ignore')


def get_image_path_list(root_path):
    image_path_list = []
    mask_path_list = []
    sub_dirs = [x[0] for x in os.walk(root_path)]
    is_root_dir = True
    for index, sub_dir in enumerate(sub_dirs):
        # print(index, sub_dir)
        if is_root_dir:
            is_root_dir = False
            continue
        # print(sub_dir, os.path.basename(sub_dir))
        files_list = os.listdir(sub_dir)
        for item in files_list:
            image_model = item.split(".")[0].strip().split("_")[-1].strip()
            if image_model != "seg":
                image_path_list.append(os.path.join(sub_dir, item))
            else:
                mask_path_list.append(os.path.join(sub_dir, item))
                mask_path_list.append(os.path.join(sub_dir, item))
                mask_path_list.append(os.path.join(sub_dir, item))
                mask_path_list.append(os.path.join(sub_dir, item))
    return image_path_list, mask_path_list


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data   # (z, x, y)


def read_img_nii(path):
    image_data_ = nib.load(path).get_data()
    return image_data_


def show_img(ori_img):
    plt.imshow(ori_img[:, :, 85], cmap='gray')  # channel_last(x, y, z)
    plt.show()


def get_mask_array(mask_slice_data, n_classes, height, width):
    """

    :param mask_slice_data:
    :param n_classes:
    :param resize:
    :param width:
    :param height:
    :return:
    """

    seg_labels = np.zeros((height, width, n_classes))

    try:

        for c in range(n_classes):
            seg_labels[:, :, c] = (mask_slice_data == c).astype(int)

    except Exception as error:  # 捕获所有可能发生的异常
        print("ERROR：", error)
    finally:
        pass
        # print("Done！")
    seg_labels = np.reshape(seg_labels, (n_classes, width * height))
    return seg_labels


def get_image_array(full_model_data, index, h, w):
    """

    :param full_model_data: model data: [t1_data, t1ce_data, t2_data, flair_data][4,240,240,155]
    :param index: The slice's index which will be selected.
    :param h: h
    :param w: w
    :return:
    """

    flair_slice = transform.resize(full_model_data[0, :, :, :][:, :, index], (h, w))
    t1_slice = transform.resize(full_model_data[0, :, :, :][:, :, index], (h, w))
    t1ce_slice = transform.resize(full_model_data[0, :, :, :][:, :, index], (h, w))
    t2_slice = transform.resize(full_model_data[0, :, :, :][:, :, index], (h, w))
    model_slice = np.array([t1_slice, t1ce_slice, t2_slice, flair_slice])
    model_slice = np.transpose(model_slice, axes=[1, 2, 0])

    return model_slice


def get_dirs_list(root_path_):
    path_list = list()
    for root, dirs, files in os.walk(root_path_):
        path_list.append(root)
    return path_list[1:]


def get_file_list(dir_name):
    files = os.listdir(dir_name)
    return files


def min_max_normalization(model_data):
    max_value = model_data.max()
    model_data = model_data/max_value
    return model_data


def image_segmentation_generator(batch_size, root_path, n_classes, output_h, output_w):
    """

    :param batch_size:
    :param root_path:
    :param n_classes:
    :param output_h:
    :param output_w:
    :return:
    """

    # patient_list = get_dirs_list(root_path)
    # 每一个病例
    while 1:
        patient_list = get_dirs_list(root_path)
        for item, value in enumerate(patient_list):
            model_data = list()

            file_list = os.listdir(value)
            file_list_copy = file_list.copy()
            mask_path = os.path.join(file_list_copy[1])
            file_list.remove(mask_path)
            mask_path = os.path.join(root_path, value, mask_path)

            for item_, value_ in enumerate(file_list):
                file_list[item_] = os.path.join(root_path, value, value_)

            mask_data = read_img_nii(mask_path)

            flair_img_data = min_max_normalization(read_img_nii(file_list[0]))
            data_shape = flair_img_data.shape
            model_data.append(flair_img_data)
            t1_img_data = min_max_normalization(read_img_nii(file_list[1]))
            model_data.append(t1_img_data)
            t1ce_img_data = min_max_normalization(read_img_nii(file_list[2]))
            model_data.append(t1ce_img_data)
            t2_img_data = min_max_normalization(read_img_nii(file_list[3]))
            model_data.append(t2_img_data)
            model_np_array = np.array(model_data)
            patient_batch = data_shape[2]//batch_size     # 向下取整
            all_slice = patient_batch * batch_size
            # slice_num = random.sample(range(len(data_shape[2])), all_slice)    #  全序列中随机取
            slice_num = random.sample(range((data_shape[2] // 2 - batch_size),
                                            (data_shape[2] // 2 + math.ceil(batch_size))), all_slice)      # 从中间随机取
            batch_data_image = list()
            batch_data_mask = list()
            # counter_batch = 0
            for index, num in enumerate(slice_num):
                if index == 0:
                    continue
                four_channel_image_data = \
                    get_image_array(model_np_array, num, output_h, output_w)   # 得到一个shape为[4, output_h, output_w]的数据
                batch_data_image.append(four_channel_image_data)
                four_channel_mask_data = \
                    transform.resize(mask_data[:, :, num], (output_h, output_w))
                # four_channel_mask_data[output_h, output_w]
                # four_channel_mask_data_v = get_mask_array(four_channel_mask_data, n_classes, output_h, output_w)
                # batch_data_mask.append(four_channel_mask_data_v)
                batch_data_mask.append([four_channel_mask_data])
                batch_data_mask_ = np.array(batch_data_mask)
                # print(batch_data_mask_.shape)
                batch_data_mask_ = np.transpose(batch_data_mask_, axes=[0, 2, 3, 1])
                if index % batch_size == 0 and index != 0:
                    # yield (np.array(batch_data_image), batch_data_mask_)
                    yield (np.array(batch_data_image).shape, batch_data_mask_.shape)


if __name__ == "__main__":
    # root_path_ = "F:\\BraTs2018\\Test_0311"
    # b_data = image_segmentation_generator(64, root_path_, 4, 255, 255)
    # a = image_segmentation_generator()
    # print(next(b_data))
    # print(b_data)
    epoch = 2
    # # root_path_ = "F:\\BraTs2018\\Test_0311"
    root_path_ = "/home/yk/Project/keras/Unet/data/HGG"
    #

    # print(b_data)
    # print(next(b_data))

    for i in range(epoch):
        b_data = image_segmentation_generator(64, root_path_, 4, 255, 255)

        print("#################")
        print("epoch{}".format(i))
        print("#################")

        for idx, (img, mask) in enumerate(b_data):
            print(img)
            print(mask)

            print("step{}".format(idx))
            print("其他计算")






