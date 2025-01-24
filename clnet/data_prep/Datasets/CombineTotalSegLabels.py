import SimpleITK as sitk
import numpy as np
import glob
import os
import csv
import shutil
from multiprocessing import Pool


class LabelMerge(object):
    def __init__(self, dict_label_idx, pth_root_input, pth_root_output, multi_thread=1):
        self.dict_label_idx = dict_label_idx
        self.subj = {}
        self.split = {}
        self.pth_out_img = {}
        self.pth_out_lab = {}
        self.multi_thread = multi_thread
        self.pth_out_img["train"] = os.path.join(pth_root_output, "imagesTr")
        if not os.path.exists(self.pth_out_img["train"]):
            os.makedirs(self.pth_out_img["train"])
        self.pth_out_img["test"] = os.path.join(pth_root_output, "imagesTs")
        if not os.path.exists(self.pth_out_img["test"]):
            os.makedirs(self.pth_out_img["test"])
        self.pth_out_img["val"] = os.path.join(pth_root_output, "imagesVal")
        if not os.path.exists(self.pth_out_img["val"]):
            os.makedirs(self.pth_out_img["val"])
        self.pth_out_lab["train"] = os.path.join(pth_root_output, "labelsTr")
        if not os.path.exists(self.pth_out_lab["train"]):
            os.makedirs(self.pth_out_lab["train"])
        self.pth_out_lab["test"] = os.path.join(pth_root_output, "labelsTs")
        if not os.path.exists(self.pth_out_lab["test"]):
            os.makedirs(self.pth_out_lab["test"])
        self.pth_out_lab["val"] = os.path.join(pth_root_output, "labelsVal")
        if not os.path.exists(self.pth_out_lab["val"]):
            os.makedirs(self.pth_out_lab["val"])
        subjects = os.listdir(pth_root_input)
        for subject in subjects:
            tmp_pth = os.path.join(pth_root_input, subject)
            if os.path.isdir(tmp_pth):
                fname_ct = os.path.join(tmp_pth, "ct.nii.gz")
                pth_seg = os.path.join(tmp_pth, "segmentations")
                if os.path.isfile(fname_ct) and os.path.isdir(pth_seg):
                    self.subj[subject] = {
                        "CT": fname_ct,
                        "Seg": glob.glob(os.path.join(pth_seg, "*.nii.gz"))
                    }
            if subject == "meta.csv":
                self._load_meta_csv(os.path.join(pth_root_input, subject))

    def _load_meta_csv(self, pth_meta_csv):
        check_split = {}
        with open(pth_meta_csv, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                row_info = row[0].split(";")
                if row_info[1] == "age":
                    continue
                if len(row_info) == 0:
                    continue
                key = row_info[0]
                self.split[key] = row_info[-1]

                if row_info[-1] not in check_split:
                    check_split[row_info[-1]] = [key]
                else:
                    check_split[row_info[-1]].append(key)
        a = 1

    def _process_single_subj(self, subj):
        try:
            # print("Processing...", subj)
            temp_img = None
            ret_dat = None
            pths_in_seg = self.subj[subj]["Seg"]
            pth_out_seg = os.path.join(self.pth_out_lab[self.split[subj]], subj + ".nii.gz")
            if not os.path.exists(pth_out_seg):
                for pth_in_seg in pths_in_seg:
                    tmp_organ = pth_in_seg.split("/")[-1][:-len(".nii.gz")]
                    tmp_img = sitk.ReadImage(pth_in_seg)
                    tmp_dat = sitk.GetArrayFromImage(tmp_img)
                    if temp_img is None:
                        temp_img = tmp_img
                    if ret_dat is None:
                        ret_dat = np.zeros(tmp_dat.shape, dtype=tmp_dat.dtype)
                    if tmp_organ in self.dict_label_idx:
                        ret_dat[tmp_dat > 0] = self.dict_label_idx[tmp_organ]
                ret_img = sitk.GetImageFromArray(ret_dat)
                ret_img.SetOrigin(temp_img.GetOrigin())
                ret_img.SetSpacing(temp_img.GetSpacing())
                ret_img.SetDirection(temp_img.GetDirection())

                sitk.WriteImage(ret_img, pth_out_seg)

                pth_in_ct = self.subj[subj]["CT"]
                pth_out_ct = os.path.join(self.pth_out_img[self.split[subj]], subj + "_0000.nii.gz")
                if not os.path.exists(pth_out_ct):
                    shutil.copy(pth_in_ct, pth_out_ct)
        except:
            print("Subject...", subj, "not readable!!!")

    def process(self):
        if self.multi_thread < 2:
            for subj in self.subj:
                self._process_single_subj(subj)
        else:
            with Pool(int(self.multi_thread)) as p:
                p.map(self._process_single_subj, self.subj)


def main():
    dict_idx_label = {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "aorta",
        8: "inferior_vena_cava",
        9: "portal_vein_and_splenic_vein",
        10: "pancreas",
        11: "adrenal_gland_right",
        12: "adrenal_gland_left",
        13: "lung_upper_lobe_left",
        14: "lung_lower_lobe_left",
        15: "lung_upper_lobe_right",
        16: "lung_middle_lobe_right",
        17: "lung_lower_lobe_right",
        18: "vertebrae_L5",
        19: "vertebrae_L4",
        20: "vertebrae_L3",
        21: "vertebrae_L2",
        22: "vertebrae_L1",
        23: "vertebrae_T12",
        24: "vertebrae_T11",
        25: "vertebrae_T10",
        26: "vertebrae_T9",
        27: "vertebrae_T8",
        28: "vertebrae_T7",
        29: "vertebrae_T6",
        30: "vertebrae_T5",
        31: "vertebrae_T4",
        32: "vertebrae_T3",
        33: "vertebrae_T2",
        34: "vertebrae_T1",
        35: "vertebrae_C7",
        36: "vertebrae_C6",
        37: "vertebrae_C5",
        38: "vertebrae_C4",
        39: "vertebrae_C3",
        40: "vertebrae_C2",
        41: "vertebrae_C1",
        42: "esophagus",
        43: "trachea",
        44: "heart_myocardium",
        45: "heart_atrium_left",
        46: "heart_ventricle_left",
        47: "heart_atrium_right",
        48: "heart_ventricle_right",
        49: "pulmonary_artery",
        50: "brain",
        51: "iliac_artery_left",
        52: "iliac_artery_right",
        53: "iliac_vena_left",
        54: "iliac_vena_right",
        55: "small_bowel",
        56: "duodenum",
        57: "colon",
        58: "rib_left_1",
        59: "rib_left_2",
        60: "rib_left_3",
        61: "rib_left_4",
        62: "rib_left_5",
        63: "rib_left_6",
        64: "rib_left_7",
        65: "rib_left_8",
        66: "rib_left_9",
        67: "rib_left_10",
        68: "rib_left_11",
        69: "rib_left_12",
        70: "rib_right_1",
        71: "rib_right_2",
        72: "rib_right_3",
        73: "rib_right_4",
        74: "rib_right_5",
        75: "rib_right_6",
        76: "rib_right_7",
        77: "rib_right_8",
        78: "rib_right_9",
        79: "rib_right_10",
        80: "rib_right_11",
        81: "rib_right_12",
        82: "humerus_left",
        83: "humerus_right",
        84: "scapula_left",
        85: "scapula_right",
        86: "clavicula_left",
        87: "clavicula_right",
        88: "femur_left",
        89: "femur_right",
        90: "hip_left",
        91: "hip_right",
        92: "sacrum",
        93: "face",
        94: "gluteus_maximus_left",
        95: "gluteus_maximus_right",
        96: "gluteus_medius_left",
        97: "gluteus_medius_right",
        98: "gluteus_minimus_left",
        99: "gluteus_minimus_right",
        100: "autochthon_left",
        101: "autochthon_right",
        102: "iliopsoas_left",
        103: "iliopsoas_right",
        104: "urinary_bladder"
    }
    dict_label_idx = {}
    for key in dict_idx_label:
        dict_label_idx[dict_idx_label[key]] = key
    pth_root_input = "/nas/dazhou.guo/Data_Partial/DataRaw/raw/TotalSeg/Totalsegmentator_dataset"
    pth_root_output = "/nas/dazhou.guo/Data_Partial/DataRaw/raw/TotalSeg/Totalsegmentator_merged_all"
    lm = LabelMerge(dict_label_idx, pth_root_input, pth_root_output, multi_thread=96)
    lm.process()


if __name__ == "__main__":
    main()
