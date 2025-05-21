from collections import OrderedDict
import os
import json
# from TotalSegList import *
import copy
from DatasetOrganIdxList import *

current_label = combine_labels(["Task003_Liver", "Task006_Lung", "Task007_Pancreas", "Task008_HepaticVessel", "Task009_Spleen", "Task010_Colon",
                                "Task017_BTCV", "Task051_StructSeg19_Task3_Thoracic_OAR", "Task055_SegTHOR", "Task062_NIHPancreas", "Task064_KiTS21",
                                "Task018_PelvicOrgan", "Task1002_FLARE22", "Task023_AMOS", "Task012_WORD", "Task501_TotalSegAbd"])
join = os.path.join

pth_root = "PATH/TO/CheckData"

task_groups = {
    # "301": "Task301_RTOG_refine_Eso",
    # "302": "Task302_RTOG_refine_BrachialPlex",
    # "303": "Task303_RTOG_refine_V_Pulmonary",
    # "700": "Task700_SuperGeneralEncoder_v3",
    # "800": "Task800_SuperGeneralEncoder_v4",
    # "012": "Task012_KiTS_GTV",
    # "013": "Task013_KiTS_Kidney",
    # "014": "Task014_LiTS",
    # "016": "Task016_StructSeg_OAR22",
    # "019": "Task019_SegTHOR_4",
    # "500": "Task500_TotalSegV2",
    # "011": "Task011_HNOAR_42",
    # "023": "Task023_AMOS",
    # "005": "Task005_ChestLNS_1_15",
    # "004": "Task004_ChestLNS_16_20",
    # "006": "Task006_Chest_EsoCancer_Organ35",
    # "009": "Task009_HNLNS_18",
    # "024": "Task024_MSD_Liver",
    # "025": "Task025_MSD_Lung",
    # "026": "Task026_MSD_Pancreas",
    # "027": "Task027_MSD_HepaticVessel",
    # "028": "Task028_MSD_Spleen",
    # "029": "Task029_MSD_Colon",
    # "034": "Task034_Liver_Ke",
    # "036": "Task036_StructSeg_NPC",
    # "031": "Task031_NPC_ZJU",
    # "030": "Task030_EsoCancer_WPY",
    # "035": "Task035_LungCancer_WPY",
    # "667": "Task667_abdomen_bone",
    # "204": "Task204_RTOG_SpinalCord",
    # "205": "Task205_RTOG_BrachialPlex",
    # "301": "Task206_RTOG_ProximalBronchi",
    # "207": "Task207_RTOG_A_Aorta",
    # "208": "Task208_RTOG_V_SVC",
    # "209": "Task209_RTOG_A_Pulmonary",
    # "210": "Task210_RTOG_V_Pulmonary",
    # "211": "Task211_RTOG_V_IVC",
    # "212": "Task212_RTOG_ChestWalls",
    # "331": "Task331_Tooth",
    "1017": "Task1017_Abdomen_ZJU_Vessel_32"
}

task_labels = {
    # "301": {1: "Eso"},
    # "302": {1: "BrachialPlex"},
    # "303": {1: "V_Pulmonary"},
    # "800": current_label,
    # "012": Task012_KiTS_GTV,
    # "013": Task013_KiTS_Kidney,
    # "014": Task014_LiTS,
    # "016": Task016_StructSeg_OAR22,
    # "019": Task019_SegTHOR_4,
    # "500": Task500_TotalSegV2,
    # "011": Task011_HNOAR_42,
    # "023": Task023_AMOS,
    # "005": Task005_ChestLNS_1_15,
    # "004": Task004_ChestLNS_16_20,
    # "006": Task006_Chest_EsoCancer_Organ35,
    # "009": Task009_HNLNS_18,
    # "024": Task024_MSD_Liver,
    # "025": Task025_MSD_Lung,
    # "026": Task026_MSD_Pancreas,
    # "027": Task027_MSD_HepaticVessel,
    # "028": Task028_MSD_Spleen,
    # "029": Task029_MSD_Colon,
    # "034": Task034_Liver_Ke,
    # "036": Task036_StructSeg_NPC,
    # "031": Task031_NPC_ZJU,
    # "030": Task030_EsoCancer_WPY,
    # "035": Task035_LungCancer_WPY,
    # "667": Task667_abdomen_bone,
    # "204": {1: "SpinalCord"},
    # "205": {1: "BrachialPlex"},
    # "206": {1: "ProximalBronchi"},
    # "207": {1: "A_Aorta"},
    # "208": {1: "V_SVC"},
    # "209": {1: "A_Pulmonary"},
    # "210": {1: "V_Pulmonary"},
    # "211": {1: "V_IVC"},
    # "212": {1: "ChestWall_L", 2: "ChestWall_R"},
    "1017": Task1017_Abdomen_ZJU_Vessel_32
}

for id in task_groups.keys():
    current_task = id
    current_json_dict = task_labels[current_task]
    current_json_dict[0] = "background"

    output_folder = join(pth_root, task_groups[current_task])
    label_dir = join(output_folder, 'labelsTr')

    train_ids = []
    test_ids = []
    train_names = os.listdir(label_dir)
    train_names.sort()
    for train_name in train_names:
        if train_name.endswith(".nii.gz"):
            train_ids.append(train_name.split('.nii.gz')[0])

    add_test_id = True
    if add_test_id:
        test_names = os.listdir(join(output_folder, 'imagesTs'))
        if not os.path.exists(join(output_folder, 'imagesTs')):
            os.makedirs(join(output_folder, 'imagesTs'))
        test_names.sort()
        for test_name in test_names:
            if test_name.endswith(".nii.gz"):
                test_ids.append(test_name.split('_0000.nii.gz')[0])

    # manually set
    json_dict = OrderedDict()
    json_dict['name'] = ""
    json_dict['description'] = task_groups[current_task]
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "private"
    json_dict['licence'] = "None"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = current_json_dict
    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
