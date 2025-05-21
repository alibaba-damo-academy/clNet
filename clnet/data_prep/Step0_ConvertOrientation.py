import SimpleITK as sitk
import numpy as np
import os
import csv
import glob
from datetime import datetime
from multiprocessing import Pool


class ReOrient(object):
    def __init__(self, pth_input, pth_output, target_ori, postfix, ftype=".nii.gz", dump_to_csv=True, multi_thread=1):
        self.table_orientation = {
            str(np.array([1, 0, 0], dtype=int)): "L",
            str(np.array([-1, 0, 0], dtype=int)): "R",
            str(np.array([0, 1, 0], dtype=int)): "P",
            str(np.array([0, -1, 0], dtype=int)): "A",
            str(np.array([0, 0, 1], dtype=int)): "S",
            str(np.array([0, 0, -1], dtype=int)): "I"
        }
        if len(target_ori) != 3:
            target_ori = target_ori[:3]
        self.target_ori = target_ori.upper()

        if ftype.startswith("."):
            self.ftype = ftype
        else:
            self.ftype = "." + ftype
        self.postfix = postfix
        self.pth_input = pth_input
        if os.path.exists(pth_input):
            if not os.path.exists(pth_output):
                os.makedirs(pth_output)
            self.pth_output = pth_output
            dt = datetime.now()
            now_date = dt.strftime("%Y%m%d")
            now_time = dt.strftime("%H%M%S")
            self.pth_output_info = os.path.join(pth_output, "info_" + now_date + "_" + now_time + ".csv")
        else:
            raise RuntimeError("The input path does not exist!!")
        self.multi_thread = multi_thread
        self.dump_to_csv = dump_to_csv

    def reori_single_image(self, pth_file):
        fname = pth_file.split("/")[-1][:-len(self.ftype)]
        # print("Processing...", fname)
        try:
            img = sitk.ReadImage(pth_file)
            ori = np.reshape(np.array(np.round(img.GetDirection()), dtype=int), (3, 3))
            ori_original = ""
            for i in range(3):
                try_key = str(ori[i])
                if try_key in self.table_orientation:
                    ori_original += self.table_orientation[try_key]
                else:
                    ori_original += "N"
            print("Processing...", fname, "from...", ori_original, "to...", self.target_ori)
            img_reori = sitk.DICOMOrient(img, self.target_ori)
            pth_output = os.path.join(self.pth_output, fname + self.postfix + ".nii.gz")
            sitk.WriteImage(img_reori, pth_output)
            if self.dump_to_csv:
                txt = [fname, ori_original, img.GetDirection(), self.target_ori,
                       img_reori.GetDirection(), pth_file, pth_output]
                with open(self.pth_output_info, "a") as out:
                    csv_out = csv.writer(out)
                    csv_out.writerow(txt)
        except:
            print("Image ...", fname, "... is not readable!")

    def reori_uniform_forward(self):
        pth_files = glob.glob(os.path.join(self.pth_input, "*" + self.ftype))
        if self.multi_thread < 2:
            for pth_file in pth_files:
                self.reori_single_image(pth_file)
        else:
            with Pool(int(self.multi_thread)) as po:
                po.map(self.reori_single_image, list(pth_files))

    def convert_single_backward(self, pth_file, ori_original, dump_to_csv):
        self.target_ori = ori_original
        self.dump_to_csv = dump_to_csv
        self.reori_single_image(pth_file)


def reori_img():
    pth_root_in = "PATH/TO/DataRaw"
    pth_root_out = "PATH/TO/clNet_raw_data/"
    tasks = [
        # "Task003_Liver/imagesTr",
        # "Task003_Liver/imagesTs",
        # "Task003_Liver/labelsTr",
        #
        # "Task006_Lung/imagesTr",
        # "Task006_Lung/imagesTs",
        # "Task006_Lung/labelsTr",
        #
        # "Task007_Pancreas/imagesTr",
        # "Task007_Pancreas/imagesTs",
        # "Task007_Pancreas/labelsTr",
        #
        # "Task008_HepaticVessel/imagesTr",
        # "Task008_HepaticVessel/imagesTs",
        # "Task008_HepaticVessel/labelsTr",
        #
        # "Task009_Spleen/imagesTr",
        # "Task009_Spleen/imagesTs",
        # "Task009_Spleen/labelsTr",
        #
        # "Task010_Colon/imagesTr",
        # "Task010_Colon/imagesTs",
        # "Task010_Colon/labelsTr",
        #
        # "Task011_AMOS/imagesTr",
        # "Task011_AMOS/imagesTs",
        # "Task011_AMOS/imagesVa",
        # "Task011_AMOS/labelsTr",
        # "Task011_AMOS/labelsVa",
        #
        # "Task012_WORD/imagesTr",
        # "Task012_WORD/imagesTs",
        # "Task012_WORD/labelsTr",
        # "Task012_WORD/labelsTs",
        #
        # "Task017_BTCV/imagesTr",
        # "Task017_BTCV/imagesTs",
        # "Task017_BTCV/labelsTr",

        # "Task018_PelvicOrgan/imagesTr",
        # "Task018_PelvicOrgan/imagesTs",
        # "Task018_PelvicOrgan/labelsTr",
        #
        # "Task050_StructSeg19_Task1_HN_OAR/imagesTr",
        # "Task050_StructSeg19_Task1_HN_OAR/labelsTr",
        #
        # "Task051_StructSeg19_Task3_Thoracic_OAR/imagesTr",
        # "Task051_StructSeg19_Task3_Thoracic_OAR/labelsTr",
        #
        # "Task055_SegTHOR/imagesTr",
        # "Task055_SegTHOR/imagesTs",
        # "Task055_SegTHOR/labelsTr",
        #
        # "Task062_NIHPancreas/imagesTr",
        # "Task062_NIHPancreas/imagesTs",
        # "Task062_NIHPancreas/labelsTr",
        #
        # "Task064_KiTS21/imagesTr",
        # "Task064_KiTS21/imagesTs",
        # "Task064_KiTS21/labelsTr",
        #
        # "Task065_KiTS21_Kidney/imagesTr",
        # "Task065_KiTS21_Kidney/imagesTs",
        # "Task065_KiTS21_Kidney/labelsTr",
        #
        # "Task1002_FLARE22/imagesTr",
        # "Task1002_FLARE22/imagesTs",
        # "Task1002_FLARE22/labelsTr",

        # "Task046_BTCV2/imagesTr/PAN",
        # "Task046_BTCV2/imagesTs",
        # "Task046_BTCV2/labelsTr",

        # "Task030_EsoCancer_WPY/imagesTr",
        # "Task030_EsoCancer_WPY/labelsTr",
        # "Task030_EsoCancer_WPY/imagesTs",
        # "Task030_EsoCancer_WPY/labelsTs",
        #
        "Task035_LungCancer_WPY/imagesTr",
        "Task035_LungCancer_WPY/labelsTr",
        "Task035_LungCancer_WPY/imagesTs",
        "Task035_LungCancer_WPY/labelsTs",

        # "Task034_Liver_Ke/imagesTr",
        # "Task034_Liver_Ke/labelsTr",
        # "Task034_Liver_Ke/imagesTs",
        # "Task034_Liver_Ke/labelsTs",

        # "Task036_StructSeg_NPC/imagesTr",
        # "Task036_StructSeg_NPC/labelsTr",
        # "Task036_StructSeg_NPC/imagesTs",
        # "Task036_StructSeg_NPC/labelsTs",

        # "Task031_NPC_ZJU/imagesTr",
        # "Task031_NPC_ZJU/labelsTr",
        # "Task031_NPC_ZJU/imagesTs",
        # "Task031_NPC_ZJU/labelsTs",

        # "Task030_EsoCancer_WPY/imagesTr",
        # "Task030_EsoCancer_WPY/labelsTr",
        # "Task030_EsoCancer_WPY/imagesTs",
        # "Task030_EsoCancer_WPY/labelsTs",



    ]
    # for task in tasks:
    #     pth_in = os.path.join(pth_root_in, task)
    #     pth_out = os.path.join(pth_root_out, task)
    #     reori = ReOrient(pth_in, pth_out, "LPS", "", multi_thread=52)
    #     reori.reori_uniform_forward()

    pth_in = "PATH/TO/CheckData/"
    pth_out = "PATH/TO/CheckData/ForWenPei_back_to_original_orientation"
    reori = ReOrient(pth_in, pth_out, "LPI", "", multi_thread=52)
    reori.reori_uniform_forward()


if __name__ == "__main__":
    reori_img()
