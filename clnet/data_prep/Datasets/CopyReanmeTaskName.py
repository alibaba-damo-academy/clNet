import os
import shutil
from multiprocessing import Pool


pth_root = "/nas/dazhou.guo/Data_Partial/DataRaw/training_dat"
pth_target = "/nas/dazhou.guo/Data_Partial/nnUNet_raw_data"

dict_order_task = {
    1: 'SuperGeneralEncoder',
    2: "AbdomenCT1K_Organ12",
    3: 'AbdomenCT1K_Organ4',
    4: 'Abdomen_Tongde_5Organ',
    5: 'ChestLNS_13',
    6: 'Chest_EsoCancer_Organ35',
    7: 'Chest_LungCancer_Organ12',
    8: 'EsoGTV_ChuanZhong',
    9: 'HNLNS_18',
    10: 'HNOAR_13',
    11: 'HNOAR_42',
    12: 'KiTS_GTV',
    13: 'KiTS_Kidney',
    14: 'LiTS',
    15: 'MultiAtlasCervix',
    16: 'StructSeg_OAR22',
    17: 'TotalSeg',
    18: 'WORD',
}

dict_task_order = {}
for o in dict_order_task:
    dict_task_order[dict_order_task[o]] = o

a =1
def copy_task(task):
    print("Copying...", task)
    task_name = "Task%03d_" % dict_task_order[task] + task
    pth_in = os.path.join(pth_root, task)
    pth_out = os.path.join(pth_target, task_name)
    shutil.copytree(pth_in, pth_out)


with Pool(20) as p:
    p.map(copy_task, dict_task_order.keys())


# for task in dict_task_order:
#     copy_task(task)