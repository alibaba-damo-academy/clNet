import os
import glob
import shutil
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool


pth_input_root = "PATH/TO/clNet_raw_data"
pth_output_root = "PATH/TO/clNet_raw_data"

# datasets = ["Task007_Chest_LungCancer_Organ12", "Task006_Chest_EsoCancer_Organ35", "Task020_StructSeg_Chest_6", "Task019_SegTHOR_4"]
datasets = ["Task801_AbdomenGE", "Task013_KITS_Kidney"]

over_write = True
if_is_train_set = True

output_task_base_id = {
    "Liver": 5001,
    "Kidneys": 5002,
    "Spleen": 5003,
    "Pancreas": 5004,
    "Aorta": 5005,
    "IVC": 5006,
    "AdrenalGlands": 5007,
    "GallBladder": 5008,
    "Esophagus": 5009,
    "Stomach": 5010,
    "Duodenum": 5011
}

task_identifier = "PL"

# input_organ_idxes_from_different_datasets = {
#     # "Eso": [1, 29, 4, 1],
#     # "Lung_L": [2, 8, 1, None],
#     # "Lung_R": [3, 9, 2, None],
#     # "Pericardium": [4, None, None, None],
#     # "SpinalCord": [5, None, None, None],
#     # "BrachialPlex": [6, None, None, None],
#     "ProximalBronchi": [7, None, None, None],
#     # "A_Aorta": [8, [1, 2, 3], None, 4],
#     # "V_SVC": [9, 18, None, None],
#     # "A_Pulmonary": [10, 4, None, None],
#     # "V_Pulmonary": [11, 35, None, None],
#     "V_IVC": [12, None, None, None],
#     # "ChestWall_L": [13, None, None, None],
#     # "ChestWall_R": [14, None, None, None]
# }

input_organ_idxes_from_different_datasets = {
    "Liver": [[1, 15, 41, 59, 69, 89], None],
    "Kidney_L": [[12, 53, 56, 71, 87], 1],
    "Kidney_R": [[11, 42, 55, 72, 86], 2],
    "Spleen": [[8, 10, 43, 54, 70, 85], None],
    "Pancreas": [[4, 20, 33, 44, 63, 76, 91], None],
    "Aorta": [[17, 32, 45], None],
    "IVC": [[18, 46, 102], None],
    "AdrenalGland_L": [[22, 48, 65, 93], None],
    "AdrenalGland_R": [[21, 47, 64, 92], None],
    "GallBladder": [[13, 49, 57, 74, 88], None],
    "Esophagus": [[14, 26, 29, 50, 58, 75, 94], None],
    "Stomach": [[16, 51, 60, 73, 90], None],
    "Duodenum": [[52, 77, 96], None],
    "Liver_GTV": [[2, 7], None],
    "Kidney_GTV": [[35, 36, 100, 101], None],
    "Pancreas_GTV": [[5], None]
}


output_organ_z_direction_image_constrain = {
    "Eso": "upper",
    "SpinalCord": "both",
    # "V_IVC": "lower",
    "A_Aorta": "lower"
}

output_organ_idxes = {
    "Liver": {"Liver": 1, "Liver_GTV": 2},
    "Kidneys": {"Kidney_L": 1, "Kidney_R": 2, "Kidney_GTV": 3},
    "Spleen": {"Spleen": 1},
    "Pancreas": {"Pancreas": 1, "Pancreas_GTV": 2},
    "Aorta": {"Aorta": 1},
    "IVC": {"IVC": 1},
    "AdrenalGlands": {"AdrenalGland_L": 1, "AdrenalGland_R": 2},
    "GallBladder": {"GallBladder": 1},
    "Esophagus": {"Esophagus": 1},
    "Stomach": {"Stomach": 1},
    "Duodenum": {"Duodenum": 1}
}

# output_organ_idxes = {
#     # "Eso": {"Eso": 1},
#     # "Lungs": {"Lung_L": 1, "Lung_R": 2},
#     # "Pericardium": {"Pericardium": 1},
#     # "SpinalCord": {"SpinalCord": 1},
#     # "BrachialPlex": {"BrachialPlex": 1},
#     "ProximalBronchi": {"ProximalBronchi": 1},
#     # "A_Aorta": {"A_Aorta": 1},
#     # "V_SVC": {"V_SVC": 1},
#     # "A_Pulmonary": {"A_Pulmonary": 1},
#     # "V_Pulmonary": {"V_Pulmonary": 1},
#     "V_IVC": {"V_IVC": 1},
#     # "ChestWalls": {"ChestWall_L": 1, "ChestWall_R": 2},
# }


class CopyImgExtractIdx(object):
    def __init__(self, output_organ, pth_ct_src, pth_ct_tar, pth_lab_src, pth_lab_tar, src_lab_idx, tar_lab_idx,  z_constrain=None,
                 prefix_ct="", postfix_ct="_0000", prefix_lab="", postfix_lab="", skip_image_without_label=False, thread=1):
        self.output_organ = output_organ

        self.pth_ct_src = pth_ct_src
        self.pth_ct_tar = pth_ct_tar
        self.prefix_ct = prefix_ct
        self.postfix_ct = postfix_ct

        self.pth_lab_src = pth_lab_src
        self.pth_lab_tar = pth_lab_tar
        self.prefix_lab = prefix_lab
        self.postfix_lab = postfix_lab

        self.src_lab_idx = src_lab_idx
        self.tar_lab_idx = tar_lab_idx

        self.skip_image_without_label = skip_image_without_label

        self.thread = thread
        if isinstance(z_constrain, dict):
            self.z_constrain = z_constrain
        else:
            self.z_constrain = {}

        self.patients = {}
        self.get_patients()

        if not os.path.exists(pth_ct_tar):
            os.makedirs(pth_ct_tar)

        if not os.path.exists(pth_lab_tar):
            os.makedirs(pth_lab_tar)

    def get_patients(self):
        if self.postfix_ct[-len(".nii.gz"):] == ".nii.gz":
            pth_src_ct_patients = glob.glob(os.path.join(self.pth_ct_src, self.prefix_ct + "*" + self.postfix_ct))
        else:
            pth_src_ct_patients = glob.glob(os.path.join(self.pth_ct_src, self.prefix_ct + "*" + self.postfix_ct + ".nii.gz"))
            self.postfix_ct += ".nii.gz"

        if self.postfix_lab[-len(".nii.gz"):] != ".nii.gz":
            self.postfix_lab += ".nii.gz"

        for pth_src_ct_patient in pth_src_ct_patients:
            fname = pth_src_ct_patient.split("/")[-1][len(self.prefix_ct):-len(self.postfix_ct)]
            pth_src_lab_patient = os.path.join(self.pth_lab_src, self.prefix_lab + fname + self.postfix_lab)
            if os.path.exists(pth_src_lab_patient):
                pth_tar_ct_patient = os.path.join(self.pth_ct_tar, self.prefix_ct + fname + self.postfix_ct)
                pth_tar_lab_patient = os.path.join(self.pth_lab_tar, self.prefix_lab + fname + self.postfix_lab)
                self.patients[fname] = {"ct": [pth_src_ct_patient, pth_tar_ct_patient], "lab": [pth_src_lab_patient, pth_tar_lab_patient]}

    def process_single_patient_label_extraction(self, fname):
        if self.src_lab_idx is not None:
            print("Processing `%s` extraction: %s" % (self.output_organ, fname))
            # pth_src_ct_patient, pth_tar_ct_patient = self.patients[fname]["ct"]
            pth_src_lab_patient, pth_tar_lab_patient = self.patients[fname]["lab"]

            src_lab_img = sitk.ReadImage(pth_src_lab_patient)
            src_lab_dat = sitk.GetArrayFromImage(src_lab_img)

            if not os.path.exists(pth_tar_lab_patient):
                tar_lab_dat = np.zeros(src_lab_dat.shape, dtype=src_lab_dat.dtype)
            else:
                tar_lab_img = sitk.ReadImage(pth_tar_lab_patient)
                tar_lab_dat = sitk.GetArrayFromImage(tar_lab_img)

            flag_save = False
            if isinstance(self.src_lab_idx, list):
                for tmp_i in self.src_lab_idx:
                    loc = np.where(src_lab_dat == tmp_i)
                    if len(loc[0]) != 0:
                        tar_lab_dat[loc] = self.tar_lab_idx
                        flag_save = True
            else:
                loc = np.where(src_lab_dat == self.src_lab_idx)
                if len(loc[0]) != 0:
                    tar_lab_dat[loc] = self.tar_lab_idx
                    flag_save = True
            if self.skip_image_without_label:
                if flag_save:
                    tar_lab_img = sitk.GetImageFromArray(tar_lab_dat)
                    tar_lab_img.CopyInformation(src_lab_img)
                    sitk.WriteImage(tar_lab_img, pth_tar_lab_patient)
            else:
                tar_lab_img = sitk.GetImageFromArray(tar_lab_dat)
                tar_lab_img.CopyInformation(src_lab_img)
                sitk.WriteImage(tar_lab_img, pth_tar_lab_patient)

    def get_img(self, dat, ref_img):
        img = sitk.GetImageFromArray(dat)
        img.SetSpacing(ref_img.GetSpacing())
        img.SetDirection(ref_img.GetDirection())
        img.SetOrigin(ref_img.GetOrigin())
        return img

    def recheck_patient_list(self):
        # self.patients[fname] = {"ct": [pth_src_ct_patient, pth_tar_ct_patient], "lab": [pth_src_lab_patient, pth_tar_lab_patient]}
        checked_patients = {}
        for patient in self.patients:
            if os.path.exists(self.patients[patient]["lab"][1]):
                checked_patients[patient] = self.patients[patient]
        self.patients = checked_patients

    def process_single_patient_z_direction_constrain(self, fname):
        if self.src_lab_idx is not None:
            print("Processing `%s` z-direction constrain: %s" % (self.output_organ, fname))
            pth_src_ct_patient, pth_tar_ct_patient = self.patients[fname]["ct"]
            _, pth_tar_lab_patient = self.patients[fname]["lab"]

            shutil.copy(pth_src_ct_patient, pth_tar_ct_patient)

            if self.output_organ in self.z_constrain:
                tar_ct_img = sitk.ReadImage(pth_tar_ct_patient)
                tar_ct_dat = sitk.GetArrayFromImage(tar_ct_img)
                tar_lab_img = sitk.ReadImage(pth_tar_lab_patient)
                tar_lab_dat = sitk.GetArrayFromImage(tar_lab_img)
                d, h, w = tar_lab_dat.shape
                location_z, _, _ = np.where(tar_lab_dat > 0)
                # Note that NifTi's z-direction is reversed
                lower_bound = int(min(location_z))
                upper_bound = min(int(max(location_z)) + 1, d)
                if "lower" in self.z_constrain[self.output_organ].lower():
                    tar_ct_dat = tar_ct_dat[lower_bound:]
                    tar_lab_dat = tar_lab_dat[lower_bound:]
                elif "upper" in self.z_constrain[self.output_organ].lower():
                    tar_ct_dat = tar_ct_dat[:upper_bound]
                    tar_lab_dat = tar_lab_dat[:upper_bound]
                elif "both" in self.z_constrain[self.output_organ].lower():
                    tar_ct_dat = tar_ct_dat[lower_bound:upper_bound]
                    tar_lab_dat = tar_lab_dat[lower_bound:upper_bound]

                ret_tar_ct_img = self.get_img(tar_ct_dat, tar_ct_img)
                sitk.WriteImage(ret_tar_ct_img, pth_tar_ct_patient)
                ret_tar_lab_img = self.get_img(tar_lab_dat, tar_lab_img)
                sitk.WriteImage(ret_tar_lab_img, pth_tar_lab_patient)

    def run_label_extraction(self):
        if self.thread < 2:
            for fname in self.patients:
                self.process_single_patient_label_extraction(fname)
        else:
            with Pool(int(self.thread)) as p:
                p.map(self.process_single_patient_label_extraction, self.patients)

    def run_z_direction_constrain(self):
        self.recheck_patient_list()
        if self.thread < 2:
            for fname in self.patients:
                self.process_single_patient_z_direction_constrain(fname)
        else:
            with Pool(int(self.thread)) as p:
                p.map(self.process_single_patient_z_direction_constrain, self.patients)


for output_organ in output_organ_idxes:
    output_task_name = "Task" + str(output_task_base_id[output_organ]) + "_" + task_identifier + "_" + output_organ
    pth_task_name = os.path.join(pth_output_root, output_task_name)
    if if_is_train_set:
        tar_pth_images = os.path.join(pth_task_name, "imagesTr")
        tar_pth_labels = os.path.join(pth_task_name, "labelsTr")
    else:
        tar_pth_images = os.path.join(pth_task_name, "imagesTs")
        tar_pth_labels = os.path.join(pth_task_name, "labelsTs")

    if over_write:
        if os.path.exists(tar_pth_images):
            shutil.rmtree(tar_pth_images)
        if os.path.exists(tar_pth_labels):
            shutil.rmtree(tar_pth_labels)

    for input_organ in output_organ_idxes[output_organ]:
        for k, input_organ_idx in enumerate(input_organ_idxes_from_different_datasets[input_organ]):
            if input_organ_idx is not None:
                if if_is_train_set:
                    src_pth_images = os.path.join(pth_input_root, datasets[k], "imagesTr")
                    src_pth_labels = os.path.join(pth_input_root, datasets[k], "labelsTr")
                else:
                    src_pth_images = os.path.join(pth_input_root, datasets[k], "imagesTs")
                    src_pth_labels = os.path.join(pth_input_root, datasets[k], "labelsTs")
                process_patient = CopyImgExtractIdx(output_organ, src_pth_images, tar_pth_images, src_pth_labels, tar_pth_labels, input_organ_idx,
                                                    output_organ_idxes[output_organ][input_organ], output_organ_z_direction_image_constrain,
                                                    skip_image_without_label=True, thread=48)
                process_patient.run_label_extraction()

for output_organ in output_organ_idxes:
    output_task_name = "Task" + str(output_task_base_id[output_organ]) + "_" + task_identifier + "_" + output_organ
    pth_task_name = os.path.join(pth_output_root, output_task_name)
    if if_is_train_set:
        tar_pth_images = os.path.join(pth_task_name, "imagesTr")
        tar_pth_labels = os.path.join(pth_task_name, "labelsTr")
    else:
        tar_pth_images = os.path.join(pth_task_name, "imagesTs")
        tar_pth_labels = os.path.join(pth_task_name, "labelsTs")

    for input_organ in output_organ_idxes[output_organ]:
        for k, input_organ_idx in enumerate(input_organ_idxes_from_different_datasets[input_organ]):
            if input_organ_idx is not None:
                if if_is_train_set:
                    src_pth_images = os.path.join(pth_input_root, datasets[k], "imagesTr")
                    src_pth_labels = os.path.join(pth_input_root, datasets[k], "labelsTr")
                else:
                    src_pth_images = os.path.join(pth_input_root, datasets[k], "imagesTs")
                    src_pth_labels = os.path.join(pth_input_root, datasets[k], "labelsTs")
                process_patient = CopyImgExtractIdx(output_organ, src_pth_images, tar_pth_images, src_pth_labels, tar_pth_labels, input_organ_idx,
                                                    output_organ_idxes[output_organ][input_organ], output_organ_z_direction_image_constrain,
                                                    skip_image_without_label=True, thread=48)
                process_patient.run_z_direction_constrain()
