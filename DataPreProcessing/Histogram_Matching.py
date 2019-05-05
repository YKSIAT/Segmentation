import SimpleITK as sitk
import numpy as np
import os
import shutil


def Histogram_Matching(inImFile, outImFile, refImFile,
                      number_of_histogram_levels=1024,
                      number_of_match_points=7,
                      threshold_at_mean_intensity=False):

    """Histogram matching of the input image with the reference image using SimpleITK
    This function uses HistogramMatchingImageFilter implemented in SimpleITK/ITK.
    Parameters
    ----------
    inImFile (string): input image file name.
    outImFile (string): output image file name: If set to 'None', the output is not saved.
    refImFile (string): reference image file name.
    number_of_histogram_levels (int): Number of histogram levels.
    number_of_match_points (int): Number of match points.
    threshold_at_mean_intensity (boolean): Threshold at mean intensity or not.
    Returns
    -------
        outputIm (SimpleITK image): histogram matched image.
    More information
    ----------------
    http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1HistogramMatchingImageFilter.html
    """
    inputIm = sitk.ReadImage(inImFile)
    referenceIm = sitk.ReadImage(refImFile)
    histMatchingFilter = sitk.HistogramMatchingImageFilter()
    histMatchingFilter.SetNumberOfHistogramLevels(number_of_histogram_levels)
    histMatchingFilter.SetNumberOfMatchPoints(number_of_match_points)
    histMatchingFilter.SetThresholdAtMeanIntensity(threshold_at_mean_intensity)
    outputIm = histMatchingFilter.Execute(inputIm, referenceIm)
    if outImFile is not None:
        sitk.WriteImage(outputIm, outImFile, True)
    return outputIm


def get_folder_lister(ori_root_path, save_root_path, refer_img_dir):
    patient_list = os.listdir(ori_root_path)
    refer_model_list = os.listdir(refer_img_dir)
    ref_flair = os.path.join(refer_img_dir, refer_model_list[0])
    ref_t1 = os.path.join(refer_img_dir, refer_model_list[2])
    ref_t1ce = os.path.join(refer_img_dir, refer_model_list[3])
    ref_t2 = os.path.join(refer_img_dir, refer_model_list[4])
    all_num = len(patient_list)
    for idx, item in enumerate(patient_list):
        modal_list = os.listdir(os.path.join(ori_root_path, item))
        save_patient_folder = os.path.join(save_root_path, item)
        if not os.path.exists(save_patient_folder):
            os.makedirs(save_patient_folder)
            print("Create folder", save_patient_folder)

        for item2 in modal_list:
            model_ab_path = os.path.join(ori_root_path, item, item2)
            save_model_ab_path = os.path.join(save_patient_folder, item2)
            # print("save_model_ab_path", save_model_ab_path)
            model_name = model_ab_path.split("\\")[-1].split(".")[0].split("_")[-1]
            # print(model_name)
            if model_name == "flair":
                Histogram_Matching(model_ab_path, save_model_ab_path, ref_flair)
                print("Processing file{} and saved to {}".format(model_ab_path, save_model_ab_path))
                pass
            elif model_name == "t1":
                Histogram_Matching(model_ab_path, save_model_ab_path, ref_t1)
                print("Processing file{} and saved to {}".format(model_ab_path, save_model_ab_path))
                pass
            elif model_name == "t1ce":
                Histogram_Matching(model_ab_path, save_model_ab_path, ref_t1ce)
                print("Processing file{} and saved to {}".format(model_ab_path, save_model_ab_path))
                pass
            elif model_name == "t2":
                Histogram_Matching(model_ab_path, save_model_ab_path, ref_t2)
                print("Processing file{} and saved to {}".format(model_ab_path, save_model_ab_path))
                pass
            else:
                shutil.copy(model_ab_path, save_model_ab_path)
                print("Copying {} to {}".format(model_ab_path, save_model_ab_path))
                pass
        print("{}/{}".format(idx+1, all_num),  " Patient {} has been successfully processed.".format(item))


def maybe_mkdir_p(directory):
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


if __name__ == "__main__":
    # input_img = r"E:\Program_new\Brats_3D\data\Brats_data\HGG\Brats18_2013_3_1\Brats18_2013_3_1_flair.nii.gz"
    # output_img = r"E:\Program_new\Brats_3D\data\Brats_data\HGG\Brats18_2013_3_1\Brats18_2013_3_1_flair_hm.nii.gz"
    # refImFile = r"E:\Program_new\Brats_3D\data\Brats_data\HGG\Brats18_2013_2_1\Brats18_2013_2_1_flair.nii.gz"
    # Histogram_Matching(refImFile, output_img, input_img)
    input_root_path = r"E:\Program_new\Brats_3D\data\Brats_data\HGG"
    ref_data_path = r"E:\Program_new\Brats_3D\data\Brats_data\HGG\Brats18_2013_2_1"
    save_path = r"E:\Program_new\Brats_3D\data\Brats_data\HGG_HM"
    get_folder_lister(input_root_path, save_path, ref_data_path)

