#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from clnet.paths import clNet_raw_data, preprocessing_output_dir, clNet_cropped_data, network_training_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


def convert_id_to_task_name(task_id: int):
    startswith = "Task%03.0d" % task_id
    if preprocessing_output_dir is not None:
        candidates_preprocessed = subdirs(preprocessing_output_dir, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if clNet_raw_data is not None:
        candidates_raw = subdirs(clNet_raw_data, prefix=startswith, join=False)
    else:
        candidates_raw = []

    if clNet_cropped_data is not None:
        candidates_cropped = subdirs(clNet_cropped_data, prefix=startswith, join=False)
    else:
        candidates_cropped = []

    candidates_trained_models = []
    if network_training_output_dir is not None:
        for m in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']:
            if isdir(join(network_training_output_dir, m)):
                candidates_trained_models += subdirs(join(network_training_output_dir, m), prefix=startswith, join=False)

    all_candidates = candidates_cropped + candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one task name found for task id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (task_id, clNet_raw_data, preprocessing_output_dir,
                                                               clNet_cropped_data))
    if len(unique_candidates) == 0:
        raise RuntimeError("Could not find a task with the ID %d." % task_id)
    return unique_candidates[0]


def convert_task_name_to_id(task_name: str):
    assert task_name.startswith("Task")
    task_name = task_name.split("_")[0]
    task_id = int(task_name[4:])
    return task_id
