import nibabel as nib
import numpy as np
import os, sys
from multiprocessing import Pool, Process, cpu_count
import SimpleITK as sitk
import matplotlib.pyplot as plt
from glob import glob
from skimage import transform
import shutil
import warnings

warnings.filterwarnings('ignore')


class CustomError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)     # 初始化父类
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo


class PreProcessData:
    def __init__(self, ori_data_path, nii_data_save_path, nii_norm_data_save_path, np_data_save_path):
        """

        :param ori_data_path:
        :param nii_data_save_path:
        :param nii_norm_data_save_path:
        :param np_data_save_path:
        """
        np.set_printoptions(precision=3)
        self.ori_data_path = ori_data_path
        self.nii_data_save_path = nii_data_save_path
        self.np_data_save_path = np_data_save_path
        self.nii_norm_data_save_path = nii_norm_data_save_path
        self.sub_file = os.listdir(ori_data_path)
        self.mkdir(self.nii_data_save_path)
        self.NUM_OF_WORKERS = cpu_count() - 1
        if self.NUM_OF_WORKERS < 1:  # in case only one processor is available, ensure that it is used
            self.NUM_OF_WORKERS = 1
    pass

    def __len__(self):
        pass

    @staticmethod
    def nii_data_read(data_path, header=True):
        """

        :param data_path: nii data path
        :param header: True or False
        :return: 3D array  # (240, 240, 155)
        """
        img = nib.load(data_path)
        image_array_data = img.get_data()
        # print("image_array_data", type(image_array_data))
        if header:
            return image_array_data, img.affine, img.header
        else:
            return image_array_data
        pass

    def process_data_2_norm(self, case_name):
        case_path = os.path.join(self.ori_data_path, case_name)
        save_case_path = os.path.join(self.nii_data_save_path, case_name)
        self.mkdir(save_case_path)
        # print("save_case_path", save_case_path)
        # print("case_path", case_path)
        nii_file_list = os.listdir(case_path)
        # print("nii_file_list", nii_file_list)

        for idx, nii in enumerate(nii_file_list):
            nii_path = os.path.join(case_path, nii)
            save_nii_path = os.path.join(save_case_path, nii)
            if nii.split(".")[0].split("_")[-1] != "seg":
                # nii_path = os.path.join(case_path, nii)
                # save_nii_path = os.path.join(save_case_path, nii)
                # print("save_nii_path", save_nii_path)
                # print("nii_path", nii_path)
                data_array, affine, _ = self.nii_data_read(nii_path)
                mask = data_array == 0
                mask_data_array = np.ma.array(data_array, mask=mask)
                mean = np.mean(mask_data_array)
                std = np.std(mask_data_array)
                norm_data = (mask_data_array - mean) / std
                # print(norm_data.data.shape)
                new_data = nib.Nifti1Image(norm_data.data, affine)
                nib.save(new_data, save_nii_path)
            else:
                try:
                    shutil.copy(nii_path, save_nii_path)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())
                pass
        pass

    def get_near_three_layers_labeled(self):
        sub_dirs = [x[0] for x in os.walk(self.nii_norm_data_save_path)]
        is_root_dir = True
        model_name = ["flair", "t1", "t1ce", "t2", "seg"]
        print("sub_dirs", sub_dirs)
        for index, sub_dir in enumerate(sub_dirs):
            if is_root_dir:
                is_root_dir = False
                continue
            # print(index, sub_dir)
            files_list = os.listdir(sub_dir)  # get all folders
            model_data = dict()
            image_save_path = os.path.join(self.np_data_save_path,
                                           os.path.basename(os.path.normpath(sub_dir)), "fusion_images")
            mask_save_path = os.path.join(self.np_data_save_path,
                                          os.path.basename(os.path.normpath(sub_dir)), "masks")
            whole_mask_save_path = os.path.join(self.np_data_save_path,
                                                os.path.basename(os.path.normpath(sub_dir)), "whole_wt")
            core_mask_save_path = os.path.join(self.np_data_save_path,
                                               os.path.basename(os.path.normpath(sub_dir)), "core_tc")
            active_mask_save_path = os.path.join(self.np_data_save_path,
                                                 os.path.basename(os.path.normpath(sub_dir)), "active_et")
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
            mask_data = self.nii_data_read(model_data[model_name[4]], header=False)
            flair_data = self.nii_data_read(model_data[model_name[0]], header=False)
            t1_data = self.nii_data_read(model_data[model_name[1]], header=False)
            t1ce_data = self.nii_data_read(model_data[model_name[2]], header=False)
            t2_data = self.nii_data_read(model_data[model_name[3]], header=False)
            # print(type(mask_data))
            shape = mask_data.shape  # nib (240, 240, 155)
            counter = 0
            for idx in range(shape[2]):
                mask_slice_data = mask_data[:, :, idx]
                if mask_slice_data.sum() != 0:
                    self.get_mask_array(mask_slice_data, regions_type="whole", all_labels=False)
                    mask_slice_data_ori = mask_slice_data
                    mask_slice_data_whole = (self.get_mask_array(mask_slice_data, regions_type="whole"))
                    mask_slice_data_core = (self.get_mask_array(mask_slice_data, regions_type="core"))
                    mask_slice_data_active = (self.get_mask_array(mask_slice_data, regions_type="active"))

                    flair_slice_data = []
                    t1_slice_data = []
                    t1ce_slice_data = []
                    t2_slice_data = []
                    if idx == 0:
                        flair_slice_data.append(flair_data[:, :, idx])
                        flair_slice_data.append(flair_data[:, :, idx])
                        flair_slice_data.append(flair_data[:, :, idx + 1])
                        flair_slice_data = np.array(flair_slice_data)

                        t1_slice_data.append(t1_data[:, :, idx])
                        t1_slice_data.append(t1_data[:, :, idx])
                        t1_slice_data.append(t1_data[:, :, idx + 1])
                        t1_slice_data = np.array(t1_slice_data)

                        t1ce_slice_data.append(t1ce_data[:, :, idx])
                        t1ce_slice_data.append(t1ce_data[:, :, idx])
                        t1ce_slice_data.append(t1ce_data[:, :, idx + 1])
                        t1ce_slice_data = np.array(t1ce_slice_data)

                        t2_slice_data.append(t2_data[:, :, idx])
                        t2_slice_data.append(t2_data[:, :, idx])
                        t2_slice_data.append(t2_data[:, :, idx + 1])
                        t2_slice_data = np.array(t2_slice_data)

                    elif idx == shape[2] - 1:
                        flair_slice_data.append(flair_data[:, :, idx - 1])
                        flair_slice_data.append(flair_data[:, :, idx])
                        flair_slice_data.append(flair_data[:, :, idx])
                        flair_slice_data = np.array(flair_slice_data)

                        t1_slice_data.append(t1_data[:, :, idx - 1])
                        t1_slice_data.append(t1_data[:, :, idx])
                        t1_slice_data.append(t1_data[:, :, idx])
                        t1_slice_data = np.array(t1_slice_data)

                        t1ce_slice_data.append(t1ce_data[:, :, idx - 1])
                        t1ce_slice_data.append(t1ce_data[:, :, idx])
                        t1ce_slice_data.append(t1ce_data[:, :, idx])
                        t1ce_slice_data = np.array(t1ce_slice_data)

                        t2_slice_data.append(t2_data[:, :, idx - 1])
                        t2_slice_data.append(t2_data[:, :, idx])
                        t2_slice_data.append(t2_data[:, :, idx])
                        t2_slice_data = np.array(t2_slice_data)
                    else:
                        flair_slice_data.append(flair_data[:, :, idx - 1])
                        flair_slice_data.append(flair_data[:, :, idx])
                        flair_slice_data.append(flair_data[:, :, idx + 1])
                        flair_slice_data = np.array(flair_slice_data)

                        t1_slice_data.append(t1_data[:, :, idx - 1])
                        t1_slice_data.append(t1_data[:, :, idx])
                        t1_slice_data.append(t1_data[:, :, idx + 1])
                        t1_slice_data = np.array(t1_slice_data)

                        t1ce_slice_data.append(t1ce_data[:, :, idx - 1])
                        t1ce_slice_data.append(t1ce_data[:, :, idx])
                        t1ce_slice_data.append(t1ce_data[:, :, idx + 1])
                        t1ce_slice_data = np.array(t1ce_slice_data)

                        t2_slice_data.append(t2_data[:, :, idx - 1])
                        t2_slice_data.append(t2_data[:, :, idx])
                        t2_slice_data.append(t2_data[:, :, idx + 1])
                        t2_slice_data = np.array(t2_slice_data)
                        # print("t2_slice_data", t2_slice_data.shape)    # (3, 240, 240)
                    data_fusion = np.concatenate((flair_slice_data, t1_slice_data, t1ce_slice_data, t2_slice_data),
                                                 axis=0)
                    # print("data_fusion.shape", data_fusion.shape)     # (12, 240, 240)
                    # mask_slice_data_ori = np.array(mask_slice_data_ori)
                    # mask_slice_data_whole = np.array(mask_slice_data_whole)
                    # mask_slice_data_core = np.array(mask_slice_data_core)
                    # mask_slice_data_active = np.array(mask_slice_data_active)
                    # print("mask_slice_data_ori", mask_slice_data_ori.shape)  # (240, 240)
                    # print("mask_slice_data_whole", mask_slice_data_whole.shape)  # (240, 240)
                    # print("mask_slice_data_core", mask_slice_data_core.shape)  # (240, 240)
                    # print("mask_slice_data_active", mask_slice_data_active.shape)  # (240, 240)

                    # print("mask_slice_data.shape", mask_slice_data.shape)    # (1, 240, 240)
                    counter += 1
                    # if counter == 20:
                    #     show_img_multiplex\
                    #         (mask_slice_data_ori, mask_slice_data_whole, mask_slice_data_core,
                    #          mask_slice_data_active, img_title="image")
                    #     break
                    # print("counter", counter)
                    # print("data_fusion", data_fusion.shape)
                    # print("mask_slice_data", mask_slice_data.shape)
                    # , os.path.basename(os.path.normpath(sub_dir)) + "_" + str(idx)
                    # print(os.path.basename(os.path.normpath(sub_dir)) + "_" + str(idx))
                    slice_name = os.path.basename(os.path.normpath(sub_dir)) + "_" + str(idx)
                    image_slice_save_path = os.path.join(image_save_path, slice_name)
                    mask_slice_save_path = os.path.join(mask_save_path, slice_name)
                    whole_mask_slice_save_path = os.path.join(whole_mask_save_path, slice_name)
                    core_mask_slice_save_path = os.path.join(core_mask_save_path, slice_name)
                    active_mask_slice_save_path = os.path.join(active_mask_save_path, slice_name)
                    # print("image_slice_save_path", image_slice_save_path)
                    # print("mask_slice_save_path", mask_slice_save_path)
                    # print("whole_mask_slice_save_path", whole_mask_slice_save_path)
                    # print("core_mask_slice_save_path", core_mask_slice_save_path)
                    # print("active_mask_slice_save_path", active_mask_slice_save_path)

                    np.save(image_slice_save_path, data_fusion)
                    np.save(mask_slice_save_path, mask_slice_data_ori)
                    np.save(whole_mask_slice_save_path, mask_slice_data_whole)
                    np.save(core_mask_slice_save_path, mask_slice_data_core)
                    np.save(active_mask_slice_save_path, mask_slice_data_active)
                    # print("sub_dir", sub_dir)
                print("{} has {} slices with tumor labels ".format(sub_dir.split("\\")[-1], counter))
                # print()

            pass

    @staticmethod
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

    def run(self):
        cases = os.listdir(self.ori_data_path)
        print("cases", cases)
        pool = Pool(self.NUM_OF_WORKERS)
        pool.map(self.process_data_2_norm, cases)


    @staticmethod
    def mkdir(path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)


if __name__ == "__main__":
    path1 = [r"H:\MICCAI19\BraTs19\Data\BraTs19Mixture_N4_HM\Train",
             r"H:\MICCAI19\BraTs19\Data\BraTs19Mixture_N4_HM\Val"]
    path2 = [r"H:\MICCAI19\BraTs19\Data\BraTs19Mixture_N4_HM_Norm\Train",
             r"H:\MICCAI19\BraTs19\Data\BraTs19Mixture_N4_HM_Norm\Val"]
    path3 = [r"H:\MICCAI19\BraTs19\Data\BraTs19Mixture_N4_HM_Norm_Layered_new\Train",
             r"H:\MICCAI19\BraTs19\Data\BraTs19Mixture_N4_HM_Norm_Layered_new\Val"]
    "ori_data_path, nii_data_save_path, nii_norm_data_save_path, np_data_save_path"
    for idx, name in enumerate(path1):
        f = PreProcessData(path1[idx], path2[idx], path2[idx], path3[idx])
        # f.process_data_2_norm(name)
        f.run()
        f.get_near_three_layers_labeled()
    pass







