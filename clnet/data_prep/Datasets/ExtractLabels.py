from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
import os
import glob
from DatasetOrganIdxList import *

source_idx_label = Task500_TotalSegV2
target_idx_label = Task501_TotalSegAbd

source_label_idx = {}
target_label_idx = {}

for idx in source_idx_label:
    source_label_idx[source_idx_label[idx]] = idx
for idx in target_idx_label:
    target_label_idx[target_idx_label[idx]] = idx


pth_source_label = "/nas/dazhou.guo/Data_Xiuan/Task500_TotalSegV2/labelsTs"
pth_target_label = "/nas/dazhou.guo/Data_Xiuan/Task501_TotalSegAbd/labelsTs"

if not os.path.exists(pth_target_label) and os.path.exists(pth_source_label):
    os.makedirs(pth_target_label)


label_files = glob.glob(os.path.join(pth_source_label, "*.nii.gz"))


def convert_label(label_file):
    fname = label_file.split("/")[-1]
    print("Converting...", fname)
    img = sitk.ReadImage(label_file)
    dat = sitk.GetArrayFromImage(img)
    dat_ret = np.zeros(dat.shape, dtype=dat.dtype)
    for label in target_label_idx:
        if label in source_label_idx:
            dat_ret[dat == source_label_idx[label]] = target_label_idx[label]
    pth_out = os.path.join(pth_target_label, fname)
    img_ret = sitk.GetImageFromArray(dat_ret)
    img_ret.CopyInformation(img)
    sitk.WriteImage(img_ret, pth_out)


if __name__ == "__main__":
    # for label_file in label_files:
    #     convert_label(label_file)

    with Pool(52) as p:
        p.map(convert_label, label_files)
