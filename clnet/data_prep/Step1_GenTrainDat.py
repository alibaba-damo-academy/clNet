#   Author @Dazhou Guo
#   Data: 07.20.2023

import shutil
from multiprocessing import Pool
import os
import glob


class GenTrainDat(object):
    def __init__(self, pth_img, pth_lab, prefix_img, prefix_lab, postfix_img, postfix_lab,
                 pth_out, train_or_not=True, ftype=".nii.gz", multi_thread=1):
        if not os.path.exists(pth_out):
            os.makedirs(pth_out)
        if train_or_not:
            folder = "Tr"
        else:
            folder = "Ts"
        self.multi_thread = multi_thread
        self.pth_out_img = os.path.join(pth_out, "imagesTr")
        if not os.path.exists(self.pth_out_img):
            os.makedirs(self.pth_out_img)
        self.pth_out_lab = os.path.join(pth_out, "labelsTr")
        if not os.path.exists(self.pth_out_lab):
            os.makedirs(self.pth_out_lab)

        self.pth_out_img = os.path.join(pth_out, "imagesTs")
        if not os.path.exists(self.pth_out_img):
            os.makedirs(self.pth_out_img)
        self.pth_out_lab = os.path.join(pth_out, "labelsTs")
        if not os.path.exists(self.pth_out_lab):
            os.makedirs(self.pth_out_lab)

        self.pth_out_img = os.path.join(pth_out, "images" + folder)
        self.pth_out_lab = os.path.join(pth_out, "labels" + folder)

        if not ftype.startswith("."):
            ftype = "." + ftype
        if ftype == postfix_img[-len(ftype):]:
            postfix_img = postfix_img[:-len(ftype)]
        if ftype == postfix_lab[-len(ftype):]:
            postfix_lab = postfix_lab[:-len(ftype)]
        imgs = glob.glob(os.path.join(pth_img, "*" + ftype))
        labs = glob.glob(os.path.join(pth_lab, "*" + ftype))

        self.dict_img = {}
        self.dict_lab = {}
        dict_img_check = {}
        for f in imgs:
            pid = f.split("/")[-1][len(prefix_img):-(len(postfix_img)+len(ftype))]
            dict_img_check[pid] = f
        for f in labs:
            pid = f.split("/")[-1][len(prefix_lab):-(len(postfix_lab)+len(ftype))]
            if pid in dict_img_check:
                self.dict_lab[pid] = f
        for pid in self.dict_lab:
            self.dict_img[pid] = dict_img_check[pid]

    def _copy_single(self, pid):
        print("Copying...", pid)
        pth_in_img = self.dict_img[pid]
        pth_in_lab = self.dict_lab[pid]

        pth_out_img = os.path.join(self.pth_out_img, pid + "_0000.nii.gz")
        pth_out_lab = os.path.join(self.pth_out_lab, pid + ".nii.gz")

        shutil.copy(pth_in_img, pth_out_img)
        shutil.copy(pth_in_lab, pth_out_lab)

    def gen(self):
        if self.multi_thread < 2:
            for pid in self.dict_img:
                self._copy_single(pid)
        else:
            with Pool(int(self.multi_thread)) as p:
                p.map(self._copy_single, self.dict_img.keys())


def main():
    pth_root = "/nas/dazhou.guo/Data_Partial/DataRaw/processed/"

    Task_name = "WORD"
    img_folder_name = "imagesTs"
    lab_folder_name = "labelsTs"

    pth_img = os.path.join(pth_root, Task_name, img_folder_name)
    pth_lab = os.path.join(pth_root, Task_name, lab_folder_name)
    prefix_img = ""
    postfix_img = ""
    prefix_lab = ""
    postfix_lab = ""
    pth_out = "/nas/dazhou.guo/Data_Partial/DataRaw/training_dat/" + Task_name

    gtd = GenTrainDat(pth_img, pth_lab, prefix_img, prefix_lab, postfix_img, postfix_lab, pth_out,
                      train_or_not=False, multi_thread=98)
    gtd.gen()


if __name__ == "__main__":
    main()
