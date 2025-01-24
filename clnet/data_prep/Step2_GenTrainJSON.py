#   Author @Dazhou Guo
#   Data: 07.20.2023

from collections import OrderedDict
import os
import json
from DatasetOrganIdxList import *

join = os.path.join

pth_root = '/nas/dazhou.guo/Data_Partial/clNet_raw_data/'

# This is the Super General Encoder JSON
task_name = "Task1004_FLARE22_no_PseudoLabel"
datasets = {
    "Task1004_FLARE22_no_PseudoLabel": Task1002_FLARE22,
}
offset = 0
label_offset = {}
for dataset in datasets:
    label_offset[dataset] = offset
    offset += len(datasets[dataset].keys())
    offset -= 1
label_dict = {}
for dataset in datasets:
    for l in datasets[dataset]:
        if l != 0:
            label_dict[l + label_offset[dataset]] = datasets[dataset][l]
label_dict[0] = "Background"

output_folder = join(pth_root, task_name)
label_dir = join(output_folder, 'labelsTr')

train_ids = []
test_ids = []
filenames = os.listdir(label_dir)
filenames.sort()
for name in filenames:
    train_ids.append(name.split('.nii.gz')[0])

add_test_id = True
if add_test_id:
    testnames = os.listdir(join(output_folder, 'imagesTs'))
    testnames.sort()
    for test_name in testnames:
        if "_0000.nii.gz" in test_name:
            test_ids.append(test_name.split('_0000.nii.gz')[0])

# manually set
json_dict = OrderedDict()
json_dict['name'] = ""
json_dict['description'] = task_name
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "private"
json_dict['licence'] = "None"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "CT",
}

json_dict['labels'] = label_dict
json_dict['numTraining'] = len(train_ids)
json_dict['numTest'] = len(test_ids)
json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]
json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
    json.dump(json_dict, f, indent=4, sort_keys=True)
