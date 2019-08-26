import numpy as np
import glob


# def data_generator(data_path, batch_size=16):
#     data_path = data_path + "/*"
#     case_list = glob.glob(data_path)
#     # print("There were {} cases".format(len(case_list)))
#     # print(len(case_list))
#     one_bath_fuse_data = []
#     one_batch_seg_data = []
#     length_case_list = len(case_list)
#     while 1:
#         for idx, case_name in enumerate(case_list):
#             one_case_data_list = glob.glob(case_name + "/*")
#             for seg_or_data in one_case_data_list:
#                 if seg_or_data.split("/")[-1].split("_")[-1] == "seg.npy":
#                     one_batch_seg_data.append(np.load(seg_or_data))
#
#                 elif seg_or_data.split("/")[-1].split("_")[-1] == "fuse.npy":
#                     one_bath_fuse_data.append(np.load(seg_or_data))
#                     # print(np.load(seg_or_data).shape)
#
#             if (idx + 1) % batch_size == 0:
#
#                 yield np.asarray(one_bath_fuse_data), np.asarray(one_batch_seg_data)
#                 one_bath_fuse_data.clear()
#                 one_batch_seg_data.clear()
#             elif idx + 1 == length_case_list:
#
#                 yield np.asarray(one_bath_fuse_data), np.asarray(one_batch_seg_data)
#                 one_bath_fuse_data.clear()
#                 one_batch_seg_data.clear()


def data_generator2(data_path, batch_size=16):
    data_path = data_path + "/*"
    case_list = glob.glob(data_path)
    print("There were {} cases".format(len(case_list)))
    # print(len(case_list))
    fuse_data = []
    seg_data = []
    batch_fuse_data = []
    batch_seg_data = []

    length_case_list = len(case_list)

    for idx, case_name in enumerate(case_list):
        one_case_data_list = glob.glob(case_name + "/*")
        for seg_or_data in one_case_data_list:
            if seg_or_data.split("/")[-1].split("_")[-1] == "seg.npy":
                seg_data.append(np.load(seg_or_data))

            elif seg_or_data.split("/")[-1].split("_")[-1] == "fuse.npy":
                fuse_data.append(np.load(seg_or_data))
                # print(np.load(seg_or_data).shape)

    while 1:
        for idx, value in enumerate(zip(fuse_data, seg_data)):
            # print(np.asarray(value[0]).shape, np.asarray(value[1]).shape)
            batch_fuse_data.append(value[0])
            batch_seg_data.append(value[1])

            if (idx + 1) % batch_size == 0:

                yield np.asarray(batch_fuse_data), np.asarray(batch_seg_data)
                batch_seg_data.clear()
                batch_fuse_data.clear()
            elif idx + 1 == length_case_list:

                yield np.asarray(batch_fuse_data), np.asarray(batch_seg_data)
                batch_fuse_data.clear()
                batch_seg_data.clear()


if __name__ == "__main__":
    # path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/Data_Aug_patch_818"
    path = r"/home/yk/Project/keras/dataset/BraTs19DataSet/New_822/data_crop_only_center"
    # gen = data_generator(path, batch_size=8)
    # for i, j in enumerate(gen):
    #     print(j[0].shape, j[1].shape)
    # data_generator(path, batch_size=8)

    # data_generator2(path)

    gen = data_generator2(path, batch_size=1)
    for i, j in enumerate(gen):
        print(j[0].shape, j[1].shape)


















