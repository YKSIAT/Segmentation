# -*- coding: UTF-8 -*-
import glob
import os
from PIL import Image
import shutil
import concurrent.futures
import SimpleITK as sitk
from nipype.interfaces.ants import N4BiasFieldCorrection


class ParallelProcessingN4BiasedField:
    def __init__(self, data_path, save_path, max_workers=30):
        self.max_workers = max_workers
        self.save_path = save_path
        self.all_patient = glob.glob(os.path.join(data_path, r"*/*"))
        self.save_path_list = [os.path.join(save_path, i.split("/")[-2], i.split("/")[-1])for i in self.all_patient]
        # print(self.save_path_list)
        # print("self.all_patient", self.all_patient)
        # self.make_path()
        # if os.path.exists(self.save_path):  
        #     shutil.rmtree(self.save_path)  
        # os.mkdir(self.save_path) 
        pass

    def __len__(self):
        return len(self.all_patient)

    def CorrectN4Bias(self, in_file, out_file):
        correct = N4BiasFieldCorrection()
        correct.inputs.input_image = in_file
        correct.inputs.output_image = out_file
        done = correct.run()
        return done.outputs.output_image

    def Processing(self, patient_file_path):
        # print("patient_file_path", patient_file_path)
        patient_file = patient_file_path.split("/")[-1]
        model_list = os.listdir(patient_file_path)
        for model_nii in model_list:
            model = model_nii.split(".")[0].split("_")[-1]
            # print("model", model)
            if model == "seg":
                # print("************************")
                output_dir = os.path.join(self.save_path,
                                          patient_file_path.split("/")[-2],
                                          patient_file_path.split("/")[-1],
                                          )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # print("output_dir", output_dir)
                print("{} is being processed...".format("{}/{}".format(patient_file, model_nii)))
                shutil.copy(os.path.join(patient_file_path, model_nii), os.path.join(self.save_path,
                                                                                     patient_file_path.split("/")[-2],
                                                                                     patient_file_path.split("/")[-1],
                                                                                     model_nii
                                                                                     ))
                print("{} is being processed...".format(patient_file))
                # print("input:{}output_Path:{}".format(os.path.join(patient_file_path, model_nii),
                #                                       os.path.join(self.save_path,
                #                                                    patient_file_path.split("/")[-2],
                #                                                    patient_file_path.split("/")[-1],
                #                                                    model_nii
                #                                                    )))
                # print(" ")
                pass
            else:
                output_dir = os.path.join(self.save_path,
                                          patient_file_path.split("/")[-2],
                                          patient_file_path.split("/")[-1],
                                          )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # print("output_dir", output_dir)
                print("{} is being processed...".format("{}/{}".format(patient_file, model_nii)))
                self.CorrectN4Bias(os.path.join(patient_file_path, model_nii),
                                   os.path.join(self.save_path,
                                   patient_file_path.split("/")[-2],
                                   patient_file_path.split("/")[-1],
                                   model_nii
                                                ))
                # print("input:{} output_Path:{}".format(os.path.join(patient_file_path, model_nii),
                #                                        os.path.join(self.save_path,
                #                                                     patient_file_path.split("/")[-2],
                #                                                     patient_file_path.split("/")[-1],
                #                                                     model_nii
                #                                                     )))
            pass

        # create and save thumbnail image
        # image = Image.open(filename)
        # image.thumbnail(size=(128, 128))
        # image.save(save_path_f, "JPEG")
        # return save_path_f

    def make_path(self):
        if os.path.exists(self.save_path): 
            shutil.rmtree(self.save_path)  
        os.mkdir(self.save_path)  

    def run(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for patient_file, out_file in zip(self.all_patient,
                                              executor.map(self.Processing, self.all_patient)):
                # print("{} is being processed to {}.".format(patient_file, out_file))
                print("{} has been processed...".format(patient_file))


if __name__ == "__main__":
    # Loop through all nii_data in the folder
    # save_path_ = r"H:\dataset\N4BiasedField"
    save_path_ = "/home/yk/Project/keras/dataset/Brats_mixture/N4BiasedField"
    # data_root_path = r"H:\dataset\BraTs"
    data_root_path = "/home/yk/Project/keras/dataset/Brats_mixture/Brats_ori"
    a = ParallelProcessingN4BiasedField(data_root_path, save_path_)
    print("A total of {} case data need to be processed.".format(len(a)))
    # a.Processing(r"H:\dataset\BraTs\HGG\Brats18_2013_2_1")
    # a.make_path()
    a.run()


