import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import transform
from keras.layers import *


import warnings

warnings.filterwarnings('ignore')


def read_img(path):
    """

    :param path:
    :return:
    """
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data   # (z, x, y)


def read_img_nii(path):
    """

    :param path:
    :return:
    """
    image_data_ = nib.load(path).get_data()
    return image_data_


def show_img(ori_img):
    plt.imshow(ori_img[:, :], cmap='gray')  # channel_last(x, y, z)
    plt.show()


def get_mask_array(mask_slice_data):
    """

    :param mask_slice_data:
    :return: mask (x, y , c)
    """
    labels = [1, 2, 4]
    n_classes = len(labels)
    mask_slice_data_shape = mask_slice_data.shape      # sk_image  (x, y, c)
    seg_labels = np.zeros((mask_slice_data_shape[1], mask_slice_data_shape[2], n_classes))

    try:

        for idx in range(n_classes):
            seg_labels[:, :, idx] = (mask_slice_data == int(labels[idx])).astype(int)

    except Exception as error:  # 捕获所有可能发生的异常
        print("ERROR：", error)
    finally:
        pass
        # print("Done！")
    # seg_labels = np.reshape(seg_labels, (width * height, n_classes))
    return seg_labels


def get_whole_mask_array(mask_slice_data):
    """

    :param mask_slice_data:(1, 240, 240)
    :return: whole tumor mask 0 1
    """
    mask_slice_data = mask_slice_data[0, :, :]
    label_num = [1, 2, 4]
    mask_slice_data_shape = mask_slice_data.shape
    seg_labels = np.zeros((mask_slice_data_shape[0], mask_slice_data_shape[1], len(label_num)))
    whole_mask = np.zeros((mask_slice_data_shape[0], mask_slice_data_shape[1]), dtype=np.uint8)

    try:

        # for c in range(n_classes):
        for idx in range(len(label_num)):
            # show_img(mask_slice_data)
            seg_labels[:, :, idx] = (mask_slice_data == int(label_num[idx])).astype(int)

        whole_mask = seg_labels[:, :, 0] + seg_labels[:, :, 1] + seg_labels[:, :, 2]

    except Exception as error:  # 捕获所有可能发生的异常
        print("ERROR：", error)
    finally:
        pass
        # print("Done！")

    # seg_labels = np.reshape(seg_labels, (width * height, n_classes))
    return whole_mask


def min_max_normalization(model_data):
    max_value = model_data.max()
    model_data = model_data/max_value
    return model_data


def standardization(model_data):
    mean = np.mean(model_data)
    std = np.std(model_data)
    result = (model_data - mean)/std
    return result


def get_image_path_list(root_path, save_path):
    image_path_list = []
    mask_path_list = []
    sub_dirs = [x[0] for x in os.walk(root_path)]
    is_root_dir = True
    model_name = ["flair", "t1", "t1ce", "t2", "seg"]
    for index, sub_dir in enumerate(sub_dirs):
        print(index, sub_dir)
        if is_root_dir:
            is_root_dir = False
            continue
        # print(sub_dir, os.path.basename(sub_dir))
        # Traverse a folder to get all data files
        files_list = os.listdir(sub_dir)   # get all folders
        model_data = dict()
        for item in files_list:
            # print("######", item, "######")
            image_model = item.split(".")[0].strip().split("_")[-1].strip()
            # print(image_model)
            if image_model == model_name[0]:
                model_data[model_name[0]] = os.path.join(sub_dir, item)
                # print(model_name[0])
                # print(model_data[model_name[0]])
            #
            elif image_model == model_name[1]:
                model_data[model_name[1]] = os.path.join(sub_dir, item)
                # print(model_name[1], model_data[model_name[1]])
            #
            elif image_model == model_name[2]:
                model_data[model_name[2]] = os.path.join(sub_dir, item)
                # print(model_name[2], model_data[model_name[2]])

            elif image_model == model_name[3]:
                model_data[model_name[3]] = os.path.join(sub_dir, item)
                # print(model_name[3], model_data[model_name[3]])

            else:
                model_data[model_name[4]] = os.path.join(sub_dir, item)
            #     print(model_name[4], model_data[model_name[4]])
            # print("hah", model_data)
        mask_data = read_img(model_data[model_name[4]])
        flair_data = read_img(model_data[model_name[0]])      # (D, X. Y)
        t1_data = read_img(model_data[model_name[1]])
        t1ce_data = read_img(model_data[model_name[2]])
        t2 = read_img(model_data[model_name[3]])
        shape = mask_data.shape
        counter = 0
        for idx in range(shape[0]):
            mask_slice_data = list()
            mask_slice_data_ = mask_data[idx, :, :]

            if mask_slice_data_.sum() != 0:

                # mask_slice_data_ = transform.resize(mask_slice_data_, (input_height, input_weight))
                mask_slice_data.append(mask_slice_data_)
                data_fusion = []
                flair_slice_data = flair_data[idx, :, :]
                # flair_slice_data = transform.resize(flair_slice_data, (input_height, input_weight))
                flair_slice_data = standardization(flair_slice_data)
                data_fusion.append(flair_slice_data)
                t1_slice_data = t1_data[idx, :, :]
                # t1_slice_data = transform.resize(t1_slice_data, (input_height, input_weight))
                t1_slice_data = standardization(t1_slice_data)
                data_fusion.append(t1_slice_data)
                t1ce_slice_data = t1ce_data[idx, :, :]
                # t1ce_slice_data = transform.resize(t1ce_slice_data, (input_height, input_weight))
                t1ce_slice_data = standardization(t1ce_slice_data)
                data_fusion.append(t1ce_slice_data)
                t2_slice_data = t2[idx, :, :]
                # t2_slice_data = transform.resize(t2_slice_data, (input_height, input_weight))
                t2_slice_data = standardization(t2_slice_data)
                data_fusion.append(t2_slice_data)
                data_fusion = np.array(data_fusion)
                mask_slice_data = np.array(mask_slice_data)
                # print(data_fusion.shape)
                counter += 1
                image_save_path = os.path.join(save_path, "image",
                                               os.path.basename(os.path.normpath(sub_dir)) + "_" + str(idx))
                print("image_save_path", image_save_path)
                mask_save_path = os.path.join(save_path, "mask",
                                              os.path.basename(os.path.normpath(sub_dir)) + "_" + str(idx))
                print("mask_save_path", mask_save_path)
                np.save(image_save_path, data_fusion)
                np.save(mask_save_path, mask_slice_data)
                print("data_fusion ", data_fusion.shape)
                print("mask_slice_data ", mask_slice_data.shape)
                # print(os.path.basename(os.path.normpath(sub_dir)))

        print("counter", counter)

    return image_path_list, mask_path_list


def generator_image_mask(data_path, batch_size):
    while True:
        sub_dirs = [x[0] for x in os.walk(data_path)]
        image_files_list = list()
        mask_files_list = list()
        for index, value in enumerate(sub_dirs):
            if index == 0:
                continue
            # print(value)

            if os.path.basename(os.path.normpath(value)) == "image":
                image_files_list = list(os.listdir(value))
            if os.path.basename(os.path.normpath(value)) == "mask":
                mask_files_list = list(os.listdir(value))
        # print("There are {} pictures and {} masks".format(len(image_files_list), len(mask_files_list)))

        assert len(image_files_list) == len(mask_files_list), \
            "The number of pictures and the number of labels must be the same"
        image_batch_data = []
        mask_batch_data = []
        for index, item in enumerate(image_files_list):
            # print(index)n

            image_data_path = os.path.join(data_path, "image", item)
            # print(image_data_path)
            # print("np.load(image_data_path).shape", np.load(image_data_path).shape)      # (4, 240, 240)
            image_batch_data.append(np.load(image_data_path))
            mask_data_path = os.path.join(data_path, "mask", item)      # (1, 240, 240)
            # print("np.load(mask_data_path).shape", np.load(mask_data_path).shape)
            # mask_batch_data.append(np.load(mask_data_path))
            mask_batch_data.append([get_whole_mask_array(np.load(mask_data_path))])
            if index == 0:
                continue
            # print(index)
            if (index + 1) % batch_size == 0:

                image_batch_data_np = np.transpose(np.array(image_batch_data), [0, 2, 3, 1])
                # get_mask_array(mask_batch_data)
                # mask_batch_data_np = np.transpose(np.array(mask_batch_data), [0, 2, 3, 1])
                # mask_batch_data_np = np.transpose(np.array(mask_batch_data), [0, 2, 3, 1])
                mask_batch_data_np = np.transpose(np.array(mask_batch_data), [0, 2, 3, 1])
                # print("mask_batch_data_np", mask_batch_data_np.shape)
                yield ([image_batch_data_np, image_batch_data_np], mask_batch_data_np)
                # yield (image_batch_data_np, mask_batch_data_np)
                image_batch_data.clear()
                mask_batch_data.clear()
            if index + 1 == len(image_files_list):
                image_batch_data_np = np.transpose(np.array(image_batch_data), [0, 2, 3, 1])
                # print("np.array(mask_batch_data)", np.array(mask_batch_data).shape)
                mask_batch_data_np = np.transpose(np.array(mask_batch_data), [0, 2, 3, 1])
                yield ([image_batch_data_np, image_batch_data_np], mask_batch_data_np)
                # yield (image_batch_data_np, mask_batch_data_np)
                image_batch_data.clear()
                mask_batch_data.clear()

                # yield (np.array(image_batch_data).shape, np.array(mask_batch_data).shape)


def generator_image_mask_val(image_path, mask_path):
    """

    :param image_path:
    :param mask_path:
    :return:
    """

    data_list = os.listdir(image_path)
    for index, value in enumerate(data_list):
        img_list = []
        mask_list = []
        image_absolute_path = os.path.join(image_path, value)
        mask_absolute_path = os.path.join(mask_path, value)
        print("image_absolute_path", image_absolute_path)
        # print("mask_absolute_path", mask_absolute_path)
        image_data = np.transpose(np.load(image_absolute_path), [1, 2, 0])
        img_list.append(image_data)
        mask_data = get_whole_mask_array(np.load(mask_absolute_path))
        mask_list.append(mask_data)
        image = np.array(img_list)
        mask = np.array(mask_list)
        yield (image, mask)
        yield ([image, image], mask)
        pass


if __name__ == "__main__":
    # data_path = "F:\\Test\\BraTs\\HGG"
    # r_path = \
    #     ["/home/yk/Project/keras/Unet/data/HGG", "/home/yk/Project/keras/Unet/data/LGG"]
    # s_path = \
    #     ["/home/yk/Project/keras/Unet/data_processed_norm/HGG_new",
    #      "/home/yk/Project/keras/Unet/data_processed_norm/LGG_new"]
    # for idx, path in enumerate(r_path):
    #     get_image_path_list(path, s_path[idx])
    # data_path = "/home/yk/Project/keras/Unet/data_processed_norm/HGG"

    # for i, value in enumerate(r_path):
    #     get_image_path_list(value, s_path[i], input_weight=256, input_height=256)
    # aa = generator_image_mask("/home/yk/Project/keras/Unet/data_processed/HGG", 32)
    # for idx, (i, j) in enumerate(aa):
    #     print(idx, i.shape, j.shape)
    #     print("other compute")

    data = generator_image_mask("/home/yk/Project/keras/Unet/data_processed_norm/LGG", 32)
    # data = generator_image_mask_val("/home/yk/Project/keras/Unet/data_processed_norm/LGG/image",
    #                                 "/home/yk/Project/keras/Unet/data_processed_norm/LGG/mask")
    for c, (i, m) in enumerate(data):
        print(i.shape)    # (1, 240, 240, 4)
        print("m.shape", m.shape)    # (1, 240, 240, 3)

        a2 = i[0, :, :, 3]
        # show_img(m[20, :, :])

        break















