import SimpleITK as sitk


class Save2Nii:
    def __init__(self, reference_path):
        """

        :param reference_path: For the first time to define class Save2Nii,you should fill in the reference path.
        like this：
        F = Save2Nii(data_path)
        F(img, filename)
        """
        self.reference_path = reference_path
        self.reference_image = sitk.ReadImage(self.reference_path)

    def __call__(self, array, save_name):
        """

        :param array:  3D array with shape (D, W, H)
        :param save_name:
        :return:
        """
        array = sitk.GetImageFromArray(array)
        array.SetSpacing(self.reference_image.GetSpacing())
        array.SetOrigin(self.reference_image.GetOrigin())
        sitk.WriteImage(array, save_name)
        print("hello")


if __name__ == "__main__":
    data_path = \
        r"F:\BraTs2018\BraTs19\MICCAI_BraTS_2019_Data_Training\HGG\BraTS19_CBICA_AAG_1\BraTS19_CBICA_AAG_1_seg.nii.gz"
    filename = r"./BraTS19_CBICA_AAG_1_seg_002.nii.gz"
    itk_img = sitk.ReadImage(data_path)
    img = sitk.GetArrayFromImage(itk_img)
    F = Save2Nii(data_path)
    F(img, filename)
