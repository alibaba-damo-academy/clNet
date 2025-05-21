import SimpleITK as sitk
import numpy as np
import os
from multiprocessing import Pool
import shutil


pth_input = "PATH/TO/DataRaw/raw/KiTS"
pth_output_img = "PATH/TO/DataRaw/raw/KiTS/images"
pth_output_lab_kidney = "PATH/TO/DataRaw/raw/KiTS/labelsKidney"
pth_output_lab_gtv = "PATH/TO/DataRaw/raw/KiTS/labelsGTV"
if not os.path.exists(pth_output_img):
    os.makedirs(pth_output_img)
if not os.path.exists(pth_output_lab_kidney):
    os.makedirs(pth_output_lab_kidney)
if not os.path.exists(pth_output_lab_gtv):
    os.makedirs(pth_output_lab_gtv)
subjects = os.listdir(pth_input)


def convert_single(subject):
    print("Processing...", subject)
    pth_ct_in = os.path.join(pth_input, subject, "imaging.nii.gz")
    pth_lab_in = os.path.join(pth_input, subject, "aggregated_OR_seg.nii.gz")
    if os.path.exists(pth_ct_in) and os.path.exists(pth_lab_in):
        pth_ct_out = os.path.join(pth_output_img, subject + "_0000.nii.gz")
        pth_lab_kidney_out = os.path.join(pth_output_lab_kidney, subject + ".nii.gz")
        pth_lab_gtv_out = os.path.join(pth_output_lab_gtv, subject + ".nii.gz")

        img = sitk.ReadImage(pth_lab_in)
        dat = sitk.GetArrayFromImage(img)

        dat_kidney = np.copy(dat)
        dat_gtv = np.copy(dat)

        dat_kidney[dat_kidney > 0] = 1
        dat_gtv[dat_gtv == 1] = 0
        dat_gtv[dat_gtv > 0] = 1

        img_kidney = sitk.GetImageFromArray(dat_kidney)
        img_gtv = sitk.GetImageFromArray(dat_gtv)

        img_kidney.SetDirection(img.GetDirection())
        img_kidney.SetSpacing(img.GetSpacing())
        img_kidney.SetOrigin(img.GetOrigin())

        img_gtv.SetDirection(img.GetDirection())
        img_gtv.SetSpacing(img.GetSpacing())
        img_gtv.SetOrigin(img.GetOrigin())

        sitk.WriteImage(img_kidney, pth_lab_kidney_out)
        sitk.WriteImage(img_gtv, pth_lab_gtv_out)
        shutil.copy(pth_ct_in, pth_ct_out)


if __name__ == "__main__":
    with Pool(96) as p:
        p.map(convert_single, subjects)
    # for subject in subjects:
    #     convert_single(subject)
