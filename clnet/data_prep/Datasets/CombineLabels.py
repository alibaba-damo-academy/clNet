import SimpleITK as sitk
import numpy as np
import glob
import os
import csv
import shutil
from multiprocessing import Pool


class LabelMerge(object):
    def __init__(self, dict_label_idx, pth_root_input, pth_root_output, prefix, postfix, multi_thread=1):
        self.dict_label_idx = dict_label_idx
        self.subj = {}
        self.split = {}
        self.pth_out_img = {}
        self.pth_out_lab = {}
        self.prefix = prefix
        self.postfix = postfix
        if not os.path.exists(pth_root_output):
            os.makedirs(pth_root_output)
        self.pth_root_output = pth_root_output
        self.multi_thread = multi_thread
        subjects = os.listdir(pth_root_input)
        for subject in subjects:
            tmp_pth = os.path.join(pth_root_input, subject)
            if os.path.isdir(tmp_pth):
                seg_files = glob.glob(os.path.join(tmp_pth, "*.nii.gz"))
                if len(seg_files) > 0:
                    self.subj[subject] = {
                        "Seg": seg_files
                    }
        a = 1

    def _process_single_subj(self, subj):
        try:
            print("Processing...", subj)
            temp_img = None
            ret_dat = None
            pths_in_seg = self.subj[subj]["Seg"]
            pth_out_seg = os.path.join(self.pth_root_output, subj + ".nii.gz")
            if not os.path.exists(pth_out_seg):
                for pth_in_seg in pths_in_seg:
                    tmp_organ = pth_in_seg.split("/")[-1][len(subj)+len(self.prefix)+1:-len(self.postfix)]
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
    dict_label_idx = {
        "Liver": 1,
        "Kidney_R": 2,
        "Spleen": 3,
        "Pancreas": 4,
        "Aorta": 5,
        "IVC": 6,
        "AdrenalGland_R": 7,
        "AdrenalGland_L": 8,
        "GallBladder": 9,
        "Esophagus": 10,
        "Stomach": 11,
        "Duodenum": 12,
        "Kidney_L": 13
    }
    dict_idx_label = {}
    for key in dict_label_idx:
        dict_idx_label[dict_label_idx[key]] = key
    pth_root_input = "PATH/TO/clNet_raw_data/Task1002_FLARE22/predsTs"
    pth_root_output = "PATH/TO/clNet_raw_data/Task1002_FLARE22/predsTs_clNet_merged"
    prefix = ""
    postfix = "_decoders.nii.gz"
    lm = LabelMerge(dict_label_idx, pth_root_input, pth_root_output, prefix, postfix, multi_thread=8)
    lm.process()


if __name__ == "__main__":
    main()
