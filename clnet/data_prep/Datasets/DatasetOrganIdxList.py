"""
Abdomen_TongDe: 56 + 5
Abdomen_CT1K: 50 + 1062 + 773(pseudo tumors)
Chest_EsoCancer: 153 + 39
Chest_LungCancer: 98
Chest_LNS: 92
EsoGTV_ChuanZhong: 1028 + 257
HNLNS_18: 21 + 4
HN_OAR_13: 244 + 62
HN_OAR_42: 112 + 30
KiTS: 300
LiTS: 262
Cervix: 30
StructSeg: 50
TotalSeg: 1405
WORD: 150

6283


NPC_GTV_SH: 260
NPC_GTV_SMU: 191
NPC_GTV_ZJU: 300

ECA_GTV: 312
EsoGTV: 148
OPX_GTV: 176

FAH_XJU: 82
FAH_ZU: 447
GPH: 50
HHA_FU: 195
SMU: 227

8671

OPX_CGMH: 38 + 58 + 91 + 62 + 27 + 57

9004

EsoGTV_CGMH: CE 352, NC 352
OPX_GTV: 176

Total: 9532
"""

Task004_Abdomen_Tongde_5Organ = {
    0: "Background",
    1: "HRCTV",
    2: "Bladder",
    3: "Rectum",
    4: "Sigmoid",
    5: "SmallBowel"
}

Task002_AbdomenCT1K_Organ12 = {
    0: "Background",
    1: "Liver",
    2: "Kidney",
    3: "Spleen",
    4: "Pancreas",
    5: "Aorta",
    6: "IVC",
    7: "Stomach",
    8: "GallBladder",
    9: "Esophagus",
    10: "AdrenalGland_R",
    11: "AdrenalGland_L",
    12: "CeliacTrunk",
}

Task003_AbdomenCT1K_Organ4 = {
    0: "Background",
    1: "Liver",
    2: "Kidney",
    3: "Spleen",
    4: "Pancreas",
}

Task006_Chest_EsoCancer_Organ35 = {
    0: "Background",
    1: "Aortic_Arch",
    10: "Spine",
    11: "Spinal_Cord",
    12: "Sternum",
    13: "Rib",
    14: "Thyroid_L",
    15: "Thyroid_R",
    16: "Trachea",
    17: "V_IVC",
    18: "V_SVC",
    19: "M_Anterior_Cervi",
    2: "Ascending_Aorta",
    20: "M_Scalenus",
    21: "M_Scalenus_Anterior",
    22: "M_Scleido",
    23: "A_CCA_L",
    24: "A_CCA_R",
    25: "A_Subclavian_L",
    26: "A_Subclavian_R",
    27: "A_Vertebral_L",
    28: "A_Vertebral_R",
    29: "Eso",
    3: "Descending_Aorta",
    30: "V_Azygos",
    31: "V_BCV_L",
    32: "V_BCV_R",
    33: "V_IJV_L",
    34: "V_IJV_R",
    35: "V_Pulmonary",
    36: "V_Subclavian_L",
    37: "V_Subclavian_R",
    38: "BrachialPlex",
    4: "A_Pulmonary",
    5: "Bronchus_L",
    6: "Bronchus_R",
    7: "Heart",
    8: "Lung_L",
    9: "Lung_R"
}

Task007_Chest_LungCancer_Organ12 = {
    0: "Background",
    1: "Eso",
    2: "Lung_L",
    3: "Lung_R",
    4: "Pericardium",
    5: "SpinalCord",
    6: "BrachialPlex",
    7: "ProximalBronchi",
    8: "A_Aorta",
    9: "V_SVC",
    10: "A_Pulmonary",
    11: "V_Pulmonary",
    12: "V_IVC",
    13: "ChestWall_L",
    14: "ChestWall_R"
}

Task005_ChestLNS_13 = {
    0: "Background",
    1: "Chest_LNS1_L",
    2: "Chest_LNS1_R",
    3: "Chest_LNS2.L",
    4: "Chest_LNS2_R",
    5: "Chest_LNS3_A",
    6: "Chest_LNS3_P",
    7: "Chest_LNS4_L",
    8: "Chest_LNS4_R",
    9: "Chest_LNS5",
    10: "Chest_LNS6",
    11: "Chest_LNS7",
    12: "Chest_LNS8",
    13: "Chest_LNS9"
}

Task008_EsoGTV_ChuanZhong = {
    0: "Background",
    1: "Eso",
    2: "Eso_GTV",
}

Task011_HNOAR_42 = {
    0: "Background",
    1: "Brachial_L",
    2: "Brachial_R",
    3: "BasalGanglia_L",
    4: "BasalGanglia_R",
    5: "BrainStem",
    6: "Cerebellum",
    7: "Cochlea_L",
    8: "Cochlea_R",
    9: "Const_I",
    10: "Const_M",
    11: "Const_S",
    12: "InnerEar_L",
    13: "InnerEar_R",
    14: "Eye_L",
    15: "Eye_R",
    16: "Epiglottis",
    17: "Eso",
    18: "Hippocampus_L",
    19: "Hippocampus_R",
    20: "HypoThalamus",
    21: "LacrimalGland_L",
    22: "LacrimalGland_R",
    23: "GSL",
    24: "Mandible_L",
    25: "Mandible_R",
    26: "OpticChiasm",
    27: "OpticNerve_L",
    28: "OpticNerve_R",
    29: "OralCavity",
    30: "Parotid_L",
    31: "Parotid_R",
    32: "PinealGland",
    33: "Pituitary",
    34: "SpineCord",
    35: "SubmandibularGland_L",
    36: "SubmandibularGland_R",
    37: "TemporalLobe_L",
    38: "TemporalLobe_R",
    39: "Thyroid_L",
    40: "Thyroid_R",
    41: "TMJoint_L",
    42: "TMJoint_R"
}

Task010_HNOAR_13 = {
    0: "Background",
    1: "BrainStem",
    2: "Eye_L",
    3: "Eye_R",
    4: "Len_L",
    5: "Len_R",
    6: "Chiasm",
    7: "OpticNerve_L",
    8: "OpticNerve_R",
    9: "Parotid_L",
    10: "Parotid_R",
    11: "TMJ_L",
    12: "TMJ_R",
    13: "SpinalCord"
}

Task009_HNLNS_18 = {
    0: "Background",
    1: "HN_LNS_1La",
    2: "HN_LNS_1Lb",
    3: "HN_LNS_1Ra",
    4: "HN_LNS_1Rb",
    5: "HN_LNS_2La",
    6: "HN_LNS_2Lb",
    7: "HN_LNS_2Ra",
    8: "HN_LNS_2Rb",
    9: "HN_LNS_3L",
    10: "HN_LNS_3R",
    11: "HN_LNS_4L",
    12: "HN_LNS_4R",
    13: "HN_LNS_5La",
    14: "HN_LNS_5Lb",
    15: "HN_LNS_5Ra",
    16: "HN_LNS_5Rb",
    17: "HN_LNS_6L",
    18: "HN_LNS_6R"
}

Task012_KiTS_GTV = {
    0: "Background",
    1: "GTV"
}

Task013_KiTS_Kidney = {
    0: "Background",
    1: "Kidney"
}

Task014_LiTS = {
    0: "Background",
    1: "Liver",
    2: "GTV"
}

Task015_MultiAtlasCervix = {
    0: "Background",
    1: "Bladder",
    2: "Uterus",
    3: "Rectum",
    4: "SmallBowel",
}

Task016_StructSeg_OAR22 = {
    0: "Background",
    1: "Brain Stem",
    2: "Eye_L",
    3: "Eye_R",
    4: "Lens_L",
    5: "Lens_R",
    6: "OpticNerve_L",
    7: "OpticNerve_R",
    8: "Chiasm",
    9: "TempLobe_L",
    10: "TempLobe_R",
    11: "Pituitary",
    12: "Parotid_L",
    13: "Parotid_R",
    14: "InnerEar_L",
    15: "InnerEar_R",
    16: "MidEar_L",
    17: "MidEar_R",
    18: "TMJ_L",
    19: "TMJ_R",
    20: "SpinalCord",
    21: "Mandible_L",
    22: "Mandible_R",
}

Task018_WORD = {
    0: "Background",
    1: "liver",
    2: "Spleen",
    3: "Kidney_L",
    4: "Kidney_R",
    5: "stomach",
    6: "Gallbladder",
    7: "Esophagus",
    8: "Pancreas",
    9: "Duodenum",
    10: "Colon",
    11: "Intestine",
    12: "Adrenal",
    13: "Rectum",
    14: "Bladder",
    15: "Head_of_femur_L",
    16: "Head_of_femur_R"
}

Task017_TotalSeg = {
    0: "Background",
    1: "Spleen",
    2: "Kidney_R",
    3: "kidney_L",
    4: "Gallbladder",
    5: "Liver",
    6: "Stomach",
    7: "Aorta",
    8: "IVC",
    9: "Portal_vein_and_Splenic_vein",
    10: "Pancreas",
    11: "AdrenalGland_R",
    12: "AdrenalGland_L",
    13: "Lung_Upper_Lobe_L",
    14: "Lung_Lower_Lobe_L",
    15: "Lung_Upper_Lobe_R",
    16: "Lung_Middle_Lobe_R",
    17: "Lung_Lower_Lobe_R",
    18: "Vertebrae_L5",
    19: "Vertebrae_L4",
    20: "Vertebrae_L3",
    21: "Vertebrae_L2",
    22: "Vertebrae_L1",
    23: "Vertebrae_T12",
    24: "Vertebrae_T11",
    25: "Vertebrae_T10",
    26: "Vertebrae_T9",
    27: "Vertebrae_T8",
    28: "Vertebrae_T7",
    29: "Vertebrae_T6",
    30: "Vertebrae_T5",
    31: "Vertebrae_T4",
    32: "Vertebrae_T3",
    33: "Vertebrae_T2",
    34: "Vertebrae_T1",
    35: "Vertebrae_C7",
    36: "Vertebrae_C6",
    37: "Vertebrae_C5",
    38: "Vertebrae_C4",
    39: "Vertebrae_C3",
    40: "Vertebrae_C2",
    41: "Vertebrae_C1",
    42: "Eso",
    43: "Trachea",
    44: "Heart_Myocardium",
    45: "Heart_Atrium_L",
    46: "Heart_ventricle_L",
    47: "Heart_atrium_R",
    48: "Heart_ventricle_R",
    49: "Pulmonary_A",
    50: "Brain",
    51: "Iliac_artery_L",
    52: "Iliac_artery_R",
    53: "Iliac_vena_L",
    54: "Iliac_vena_R",
    55: "Small_bowel",
    56: "Duodenum",
    57: "Colon",
    58: "Rib_L_1",
    59: "Rib_L_2",
    60: "Rib_L_3",
    61: "Rib_L_4",
    62: "Rib_L_5",
    63: "Rib_L_6",
    64: "Rib_L_7",
    65: "Rib_L_8",
    66: "Rib_L_9",
    67: "Rib_L_10",
    68: "Rib_L_11",
    69: "Rib_L_12",
    70: "Rib_R_1",
    71: "Rib_R_2",
    72: "Rib_R_3",
    73: "Rib_R_4",
    74: "Rib_R_5",
    75: "Rib_R_6",
    76: "Rib_R_7",
    77: "Rib_R_8",
    78: "Rib_R_9",
    79: "Rib_R_10",
    80: "Rib_R_11",
    81: "Rib_R_12",
    82: "Humerus_L",
    83: "Humerus_R",
    84: "Scapula_L",
    85: "Scapula_R",
    86: "Clavicula_L",
    87: "Clavicula_R",
    88: "Femur_L",
    89: "Femur_R",
    90: "Hip_L",
    91: "Hip_R",
    92: "Sacrum",
    93: "Face",
    94: "Gluteus_maximus_L",
    95: "Gluteus_maximus_R",
    96: "Gluteus_medius_L",
    97: "Gluteus_medius_R",
    98: "Gluteus_minimus_L",
    99: "Gluteus_minimus_R",
    100: "Autochthon_L",
    101: "Autochthon_R",
    102: "Iliopsoas_L",
    103: "Iliopsoas_R",
    104: "Urinary_bladder"
}

Task500_TotalSegV2 = {
    0: "Background",
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_right_4",
    93: "rib_right_3",
    94: "rib_left_1",
    95: "rib_left_2",
    96: "rib_left_3",
    97: "rib_left_4",
    98: "rib_left_5",
    99: "rib_left_6",
    100: "rib_left_7",
    101: "rib_left_8",
    102: "rib_left_9",
    103: "rib_left_10",
    104: "rib_left_11",
    105: "rib_left_12",
    106: "rib_right_1",
    107: "rib_right_2",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages"
}

# XiuAn's Data
Task003_Liver = {
    0: "Background",
    1: "Liver",
    2: "Liver_GTV",
}

Task006_Lung = {
    0: "Background",
    1: "Lung_GTV",
}

Task007_Pancreas = {
    0: "Background",
    1: "Pancreas",
    2: "Pancreas_GTV",
}

Task008_HepaticVessel = {
    0: "Background",
    1: "HepaticVessel",
    2: "Liver_GTV",
}

Task009_Spleen = {
    0: "Background",
    1: "Spleen",
}

Task010_Colon = {
    0: "Background",
    1: "Colon_GTV",
}

Task017_BTCV = {
    0: "Background",
    1: "Spleen",
    2: "Kidney_R",
    3: "Kidney_L",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "IVC",
    10: "PortalVein_SplenicVein",
    11: "Pancreas",
    12: "AdrenalGland_R",
    13: "AdrenalGland_L"
}

Task046_BTCV2 = {
    0: "Background",
    1: "Spleen",
    2: "left kidney",
    3: "gallbladder",
    4: "esophagus",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "duodenum"
}

Task050_StructSeg19_Task1_HN_OAR = {
    0: "Background",
    1: "BrainStem",
    2: "Eye_L",
    3: "Eye_R",
    4: "Lens_L",
    5: "Lens_R",
    6: "OpticNerve_L",
    7: "OpticNerve_R",
    8: "Chiasm",
    9: "TempLobe_L",
    10: "TempLobe_R",
    11: "Pituitary",
    12: "Parotid_L",
    13: "Parotid_R",
    14: "InnerEar_L",
    15: "InnerEar_R",
    16: "MidEar_L",
    17: "MidEar_R",
    18: "TMJ_L",
    19: "TMJ_R",
    20: "SpinalCord",
    21: "Mandible_L",
    22: "Mandible_R"
}

Task051_StructSeg19_Task3_Thoracic_OAR = {
    0: "Background",
    1: "Lung_L",
    2: "Lung_R",
    3: "Heart",
    4: "Esophagus",
    5: "Trachea",
    6: "SpinalCord"
}

Task055_SegTHOR = {
    0: "Background",
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta"
}

Task062_NIHPancreas = {
    0: "Background",
    1: "Pancreas"
}

Task064_KiTS21 = {
    0: "Background",
    1: "Kidney",
    2: "Kidney_GTV",
    3: "Kidney_Cyst"
}

Task018_PelvicOrgan = {
    0: "Background",
    1: "Bladder",
    2: "Uterus",
    3: "Rectum",
    4: "SmallBowel"
}

Task1002_FLARE22 = {
    0: "Background",
    1: "Liver",
    2: "Kidney_R",
    3: "Spleen",
    4: "Pancreas",
    5: "Aorta",
    6: "IVC",
    7: "AdrenalGland_R",
    8: "AdrenalGland_L",
    9: "GallBladder",
    10: "Esophagus",
    11: "Stomach",
    12: "Duodenum",
    13: "Kidney_L"
}

Task011_AMOS = {
    0: "Background",
    1: "Spleen",
    2: "Kidney_R",
    3: "Kidney_L",
    4: "GallBladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Arota",
    9: "PostCava",
    10: "Pancreas",
    11: "AdrenalGland_R",
    12: "AdrenalGland_L",
    13: "Duodenum",
    14: "Bladder",
    15: "Prostate/Uterus",
}

Task012_WORD = {
    0: "Background",
    1: "liver",
    2: "Spleen",
    3: "Kidney_L",
    4: "Kidney_R",
    5: "Stomach",
    6: "Gallbladder",
    7: "Esophagus",
    8: "Pancreas",
    9: "Duodenum",
    10: "Colon",
    11: "Intestine",
    12: "Adrenal",
    13: "Rectum",
    14: "Bladder",
    15: "Head_of_femur_L",
    16: "Head_of_femur_R"
}

Task501_TotalSegAbd = {
    0: "Background",
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "esophagus",
    11: "small_bowel",
    12: "duodenum",
    13: "colon",
    14: "urinary_bladder",
    15: "prostate",
    16: "kidney_cyst_left",
    17: "kidney_cyst_right",
    18: "inferior_vena_cava",
}


Task667_abdomen_bone = {
    0: "Background",
    1: "humerus_left",
    2: "humerus_right",
    3: "scapula_left",
    4: "scapula_right",
    5: "clavicula_left",
    6: "clavicula_right",
    7: "femur_left",
    8: "femur_right",
    9: "hip_left",
    10: "hip_right",
    11: "skull",
    12: "sacrum",
    13: "vertebrae_S1",
    14: "vertebrae_L5",
    15: "vertebrae_L4",
    16: "vertebrae_L3",
    17: "vertebrae_L2",
    18: "vertebrae_L1",
    19: "vertebrae_T12",
    20: "vertebrae_T11",
    21: "vertebrae_T10",
    22: "vertebrae_T9",
    23: "vertebrae_T8",
    24: "vertebrae_T7",
    25: "vertebrae_T6",
    26: "vertebrae_T5",
    27: "vertebrae_T4",
    28: "vertebrae_T3",
    29: "vertebrae_T2",
    30: "vertebrae_T1",
    31: "vertebrae_C7",
    32: "vertebrae_C6",
    33: "vertebrae_C5",
    34: "vertebrae_C4",
    35: "vertebrae_C3",
    36: "vertebrae_C2",
    37: "vertebrae_C1",
    38: "rib_left_1",
    39: "rib_left_2",
    40: "rib_left_3",
    41: "rib_left_4",
    42: "rib_left_5",
    43: "rib_left_6",
    44: "rib_left_7",
    45: "rib_left_8",
    46: "rib_left_9",
    47: "rib_left_10",
    48: "rib_left_11",
    49: "rib_left_12",
    50: "rib_right_1",
    51: "rib_right_2",
    52: "rib_right_3",
    53: "rib_right_4",
    54: "rib_right_5",
    55: "rib_right_6",
    56: "rib_right_7",
    57: "rib_right_8",
    58: "rib_right_9",
    59: "rib_right_10",
    60: "rib_right_11",
    61: "rib_right_12",
    62: "sternum",
    63: "costal_cartilages"
}


def combine_labels(list_of_datasets):
    ret_label = {}
    offset = 0
    for dataset in list_of_datasets:
        for label in dataset:
            if label == 0:
                offset -= 1
                continue
            ret_label[label + offset] = dataset[label]
        offset += len(dataset)
    return ret_label
