from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
import shutil
import os
import glob
import filecmp
from DatasetOrganIdxList import *


class CombineDataset(object):
    def __init__(self, pth_root, pth_out, datasets, dict_label):
        self.imagesTr = {}
        self.imagesTs = {}
        self.labelsTr = {}
        self.labelsTs = {}
        self.pth_out = pth_out
        self.datasets = datasets
        self.label_offset = {}
        offset = 0
        for dataset in dict_label:
            self.label_offset[dataset] = offset
            offset += len(dict_label[dataset].keys())
            offset -= 1
        for dataset in datasets:
            imagesTr = glob.glob(os.path.join(pth_root, dataset, "imagesTr", "*.nii.gz"))
            for img in imagesTr:
                self.imagesTr[img] = dataset
            imagesTs = glob.glob(os.path.join(pth_root, dataset, "imagesTs", "*.nii.gz"))
            for img in imagesTs:
                self.imagesTs[img] = dataset
            labelsTr = glob.glob(os.path.join(pth_root, dataset, "labelsTr", "*.nii.gz"))
            for lab in labelsTr:
                self.labelsTr[lab] = dataset
            labelsTs = glob.glob(os.path.join(pth_root, dataset, "labelsTs", "*.nii.gz"))
            for lab in labelsTs:
                self.labelsTs[lab] = dataset
        a = 1

    def copy_single_imageTr(self, pth_in):
        print("Copying...", pth_in)
        fname = pth_in.split("/")[-1]
        dataset = self.imagesTr[pth_in].split("/")[-1]
        fname = dataset + "_" + fname
        pth_out = os.path.join(self.pth_out, "imagesTr", fname)
        if not os.path.exists(pth_out):
            shutil.copy(pth_in, pth_out)

    def copy_single_imageTs(self, pth_in):
        print("Copying...", pth_in)
        fname = pth_in.split("/")[-1]
        dataset = self.imagesTs[pth_in].split("/")[-1]
        fname = dataset + "_" + fname
        pth_out = os.path.join(self.pth_out, "imagesTs", fname)
        if not os.path.exists(pth_out):
            shutil.copy(pth_in, pth_out)

    def copy(self):
        with Pool(52) as p:
            p.map(self.copy_single_imageTr, self.imagesTr.keys())
            p.map(self.copy_single_imageTs, self.imagesTs.keys())
        # for pth in self.imagesTr.keys():
        #     self.copy_single_imageTr(pth)

    def combine_single_labelTr(self, pth_in):
        # print("Converting...", pth_in)
        fname = pth_in.split("/")[-1]
        dataset = self.labelsTr[pth_in]
        img = sitk.ReadImage(pth_in)
        dat = sitk.GetArrayFromImage(img).astype(np.uint16)
        dat += self.label_offset[dataset]
        dat[dat == self.label_offset[dataset]] = 0
        print("Converting...", pth_in, np.unique(dat))
        img_ret = sitk.GetImageFromArray(dat)
        img_ret.SetDirection(img.GetDirection())
        img_ret.SetSpacing(img.GetSpacing())
        img_ret.SetOrigin(img.GetOrigin())
        fname = dataset + "_" + fname
        pth_out = os.path.join(self.pth_out, "labelsTr", fname)
        sitk.WriteImage(img_ret, pth_out)

    def combine_single_labelTs(self, pth_in):
        # print("Converting...", pth_in)
        fname = pth_in.split("/")[-1]
        dataset = self.labelsTs[pth_in]
        img = sitk.ReadImage(pth_in)
        dat = sitk.GetArrayFromImage(img).astype(np.uint16)
        dat += self.label_offset[dataset]
        dat[dat == self.label_offset[dataset]] = 0
        print("Converting...", pth_in, np.unique(dat))
        img_ret = sitk.GetImageFromArray(dat)
        img_ret.SetDirection(img.GetDirection())
        img_ret.SetSpacing(img.GetSpacing())
        img_ret.SetOrigin(img.GetOrigin())
        fname = dataset + "_" + fname
        pth_out = os.path.join(self.pth_out, "labelsTs", fname)
        sitk.WriteImage(img_ret, pth_out)

    def combine(self):
        with Pool(52) as p:
            p.map(self.combine_single_labelTr, self.labelsTr.keys())
            # p.map(self.combine_single_labelTs, self.labelsTs.keys())
        # for pth_in in self.labelsTr:
        #     self.combine_single_labelTr(pth_in)

dict_label = {
    "Task003_Liver": Task003_Liver,
    "Task006_Lung": Task006_Lung,
    "Task007_Pancreas": Task007_Pancreas,
    "Task008_HepaticVessel": Task008_HepaticVessel,
    "Task009_Spleen": Task009_Spleen,
    "Task010_Colon": Task010_Colon,
    "Task017_BTCV": Task017_BTCV,
    "Task051_StructSeg19_Task3_Thoracic_OAR": Task051_StructSeg19_Task3_Thoracic_OAR,
    "Task055_SegTHOR": Task055_SegTHOR,
    "Task062_NIHPancreas": Task062_NIHPancreas,
    "Task064_KiTS21": Task064_KiTS21,
    "Task018_PelvicOrgan": Task018_PelvicOrgan,
    "Task1002_FLARE22": Task1002_FLARE22,
    "Task011_AMOS": Task011_AMOS,
    "Task012_WORD": Task012_WORD,
    "Task501_TotalSegAbd": Task501_TotalSegAbd,
}


def main():
    datasets = ["Task003_Liver", "Task006_Lung", "Task007_Pancreas", "Task008_HepaticVessel", "Task009_Spleen", "Task010_Colon",
                "Task017_BTCV", "Task051_StructSeg19_Task3_Thoracic_OAR", "Task055_SegTHOR", "Task062_NIHPancreas", "Task064_KiTS21",
                "Task018_PelvicOrgan", "Task1002_FLARE22", "Task011_AMOS", "Task012_WORD", "Task501_TotalSegAbd"]
    pth_root = "PATH/TO/reoriented/"
    # pth_out = "PATH/TO/nnUNet_raw_data/Task501_SuperGeneralEncoder_HN"
    # pth_out = "PATH/TO/nnUNet_raw_data/Task502_SuperGeneralEncoder_Chest"
    # pth_out = "PATH/TO/nnUNet_raw_data/Task503_SuperGeneralEncoder_Abdomen"
    # pth_out = "PATH/TO/nnUNet_raw_data/Task500_SuperGeneralEncoder_HN"
    pth_out = "PATH/TO/reoriented/Task801_AbdomenGE"
    if not os.path.exists(os.path.join(pth_out, "imagesTr")):
        os.makedirs(os.path.join(pth_out, "imagesTr"))
    if not os.path.exists(os.path.join(pth_out, "imagesTs")):
        os.makedirs(os.path.join(pth_out, "imagesTs"))
    if not os.path.exists(os.path.join(pth_out, "labelsTr")):
        os.makedirs(os.path.join(pth_out, "labelsTr"))
    if not os.path.exists(os.path.join(pth_out, "labelsTs")):
        os.makedirs(os.path.join(pth_out, "labelsTs"))
    cd = CombineDataset(pth_root, pth_out, datasets, dict_label)
    # cd.copy()
    cd.combine()


if __name__ == "__main__":
    main()
