#   Author @Dazhou Guo
#   Data: 07.12.2023

import json
import os
import pickle
import shutil
from collections import OrderedDict
from multiprocessing import Pool
import SimpleITK as sitk
import warnings

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, \
    subdirs, isfile, save_json
from clnet.configuration import default_num_threads
from clnet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from clnet.experiment_planning.common_utils import split_4d_nifti
from clnet.paths import clNet_raw_data, clNet_cropped_data, preprocessing_output_dir
from clnet.preprocessing.cropping import ImageCropper
# import time


def split_4d(input_folder, num_processes=default_num_threads, overwrite_task_output_id=None):
    assert isdir(join(input_folder, "imagesTr")) and isdir(join(input_folder, "labelsTr")) and \
           isfile(join(input_folder, "dataset.json")), \
        "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the " \
        "imagesTr and labelsTr subfolders and the dataset.json file"

    while input_folder.endswith("/"):
        input_folder = input_folder[:-1]

    full_task_name = input_folder.split("/")[-1]

    assert full_task_name.startswith("Task"), "The input folder must point to a folder that starts with TaskXX_"

    first_underscore = full_task_name.find("_")
    assert first_underscore == 6, "Input folder start with TaskXX with XX being a 3-digit id: 00, 01, 02 etc"

    input_task_id = int(full_task_name[4:6])
    if overwrite_task_output_id is None:
        overwrite_task_output_id = input_task_id

    task_name = full_task_name[7:]

    output_folder = join(clNet_raw_data, "Task%03.0d_" % overwrite_task_output_id + task_name)

    if isdir(output_folder):
        shutil.rmtree(output_folder)

    files = []
    output_dirs = []

    maybe_mkdir_p(output_folder)
    for subdir in ["imagesTr", "imagesTs"]:
        curr_out_dir = join(output_folder, subdir)
        if not isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = join(input_folder, subdir)
        nii_files = [join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    shutil.copytree(join(input_folder, "labelsTr"), join(output_folder, "labelsTr"))

    p = Pool(num_processes)
    p.starmap(split_4d_nifti, zip(files, output_dirs))
    p.close()
    p.join()
    shutil.copy(join(input_folder, "dataset.json"), output_folder)


def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d["training"]
    num_modalities = len(d["modality"].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr["image"].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(join(base_folder_splitted, "labelsTr", tr["label"].split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d["modality"][str(i)] for i in d["modality"].keys()}


def create_lists_from_splitted_dataset_folder(folder):
    """
    does not rely on dataset.json
    :param folder:
    :return:
    """
    caseIDs = get_caseIDs_from_splitted_dataset_folder(folder)
    list_of_lists = []
    for f in caseIDs:
        list_of_lists.append(subfiles(folder, prefix=f, suffix=".nii.gz", join=True, sort=True))
    return list_of_lists


def get_caseIDs_from_splitted_dataset_folder(folder):
    files = subfiles(folder, suffix=".nii.gz", join=False)
    # all files must be .nii.gz and have 4 digit modality index
    files = [i[:-12] for i in files]
    # only unique patient ids
    files = np.unique(files)
    return files


def collect_bpr_scores_per_label(task_name, classes, overwrite=True):
    if os.path.isfile(join(clNet_cropped_data, task_name, "bpr_scores.json")) and not overwrite:
        from batchgenerators.utilities.file_and_folder_operations import load_json
        bpr_json = load_json(join(clNet_cropped_data, task_name, "bpr_scores.json"))
        return bpr_json
    bpr_json = OrderedDict()
    bpr_json_per_label = OrderedDict()
    for key in classes:
        if key == "0":
            continue
        bpr_json[key] = OrderedDict()
        bpr_json[key] = {"name": classes[key], "median": np.nan, "mean": [], "sd": np.nan,
                         "mn": np.nan, "mx": np.nan, "percentile_99_5": [], "percentile_00_5": []}
        bpr_json_per_label[key] = np.asarray([])

    list_of_bpr_files = sorted(subfiles(join(clNet_cropped_data, task_name), True, None, "_0000.json", True))

    collect_bpr_scores_per_label_per_file = CollectBPRScoresPerLabelPerFile(list_of_bpr_files, task_name, classes)
    collect_bpr_scores_per_label_per_file.run()

    for bpr_file in list_of_bpr_files:
        with open(collect_bpr_scores_per_label_per_file.output_fnames[bpr_file], "rb") as f_pickle:
            bpr_json_per_label = pickle.load(f_pickle)
            # Calculate stats for each class
            for key in classes:
                if int(key) == 0:
                    continue
                scores = bpr_json_per_label[key]
                # reset the "bpr_json_per_label" to empty
                bpr_json_per_label[key] = []
                if len(scores) != 0:
                    scores = scores[~(np.isnan(scores))]
                    # print("{} total slices: {}".format(key, len(scores)))

                    if len(scores) != 0:
                        mn = np.min(scores)
                        mx = np.max(scores)
                        percentile_99_5 = np.percentile(scores, 99.5)
                        percentile_00_5 = np.percentile(scores, 00.5)
                        mean = mx - mn

                        if np.isnan(bpr_json[key]["mn"]):
                            bpr_json[key]["mn"] = mn
                        else:
                            bpr_json[key]["mn"] = min(bpr_json[key]["mn"], mn)

                        if np.isnan(bpr_json[key]["mx"]):
                            bpr_json[key]["mx"] = mx
                        else:
                            bpr_json[key]["mx"] = max(bpr_json[key]["mx"], mx)

                        bpr_json[key]["percentile_99_5"].append(percentile_99_5)
                        bpr_json[key]["percentile_00_5"].append(percentile_00_5)
                        bpr_json[key]["mean"].append(mean)

    for key in bpr_json:
        if len(bpr_json[key]["mean"]) > 0:
            bpr_json[key]["sd"] = np.std(bpr_json[key]["mean"])
            bpr_json[key]["median"] = np.median(bpr_json[key]["mean"])
            bpr_json[key]["mean"] = np.mean(bpr_json[key]["mean"])
            bpr_json[key]["percentile_99_5"] = np.mean(bpr_json[key]["percentile_99_5"])
            bpr_json[key]["percentile_00_5"] = np.mean(bpr_json[key]["percentile_00_5"])
        else:
            bpr_json[key]["sd"] = np.nan
            bpr_json[key]["median"] = np.nan
            bpr_json[key]["mean"] = np.nan
            bpr_json[key]["percentile_99_5"] = np.nan
            bpr_json[key]["percentile_00_5"] = np.nan

    save_json(bpr_json, join(clNet_cropped_data, task_name, "bpr_scores.json"), sort_keys=False)
    print(bpr_json)
    return bpr_json


class CollectBPRScoresPerLabelPerFile(object):
    def __init__(self, list_of_bpr_files, task_name, classes, over_write=True, num_theads=default_num_threads):
        self.task_name = task_name
        self.classes = classes
        self.num_theads = num_theads
        self.list_of_bpr_files = list_of_bpr_files
        self.output_fnames = {}
        self.over_write = over_write
        for bpr_file in list_of_bpr_files:
            output_fname = bpr_file.split("/")[-1].replace("_0000.json", "_bpr_json_per_label.pkl")
            self.output_fnames[bpr_file] = join(clNet_cropped_data, self.task_name, output_fname)

    def run(self):
        if self.num_theads < 2:
            for bpr_file in self.list_of_bpr_files:
                self._process_single_file(bpr_file)
        else:
            with Pool(self.num_theads) as p:
                p.map(self._process_single_file, self.list_of_bpr_files)

    def _process_single_file(self, bpr_file):
        if os.path.isfile(self.output_fnames[bpr_file]) and not self.over_write:
            print("File {} already exists, skipping".format(self.output_fnames[bpr_file]))
        else:
            bpr_json_per_label = {}
            seg_file = join(clNet_raw_data, self.task_name, "labelsTr", bpr_file.split("/")[-1].replace("_0000.json", ".nii.gz"))
            seg = sitk.ReadImage(seg_file)
            seg_data = sitk.GetArrayFromImage(seg)
            bpr_json_raw = json.load(open(bpr_file))
            bpr_scores = np.asarray(bpr_json_raw["cleaned slice scores"])
            mean_interval = np.nanmean(np.diff(bpr_scores))
            # it is possible that bpr_json_raw["cleaned slice scores"] are all NaN
            # we use bpr_json_raw["unprocessed slice scores"] instead
            if np.isnan(bpr_scores).all():
                bpr_scores = np.asarray(bpr_json_raw["unprocessed slice scores"])
            if np.isnan(mean_interval):
                interval_all = np.diff(np.asarray(bpr_json_raw["unprocessed slice scores"]))
                # try to remove nan from "interval_all"
                interval_all = interval_all[~np.isnan(interval_all)]
                # try to set up the lower and upper bound -> in case of spikes or valleys
                interval_bound_low, interval_bound_up = np.percentile(interval_all, 00.5), np.percentile(interval_all, 99.5)
                # clip the "interval_all" using the low/up bounds
                interval_all = np.clip(interval_all, a_min=interval_bound_low, a_max=interval_bound_up)
                if len(interval_all) < 3:
                    warnings.warn("Cannot get body parts using given input image %s" % seg_file)
                mean_interval = np.nanmean(interval_all)
            for i, s in enumerate(bpr_scores):
                if i == 0:
                    continue
                if (np.isnan(s)) and (not np.isnan(bpr_scores[i - 1])):
                    bpr_scores[i] = bpr_scores[i - 1] + mean_interval

            for i, s in enumerate(bpr_scores[::-1]):
                i = len(bpr_scores) - 1 - i
                if (np.isnan(s)) and (not np.isnan(bpr_scores[i + 1])):
                    bpr_scores[i] = bpr_scores[i + 1] - mean_interval
            bpr_json_raw["cleaned slice scores"] = list(bpr_scores)
            assert not np.isnan(bpr_scores).any()
            save_json(bpr_json_raw, bpr_file, sort_keys=False)
            for key in self.classes:
                label_id = int(key)
                if label_id == 0:
                    continue
                bpr_scores_per_label = bpr_scores[(seg_data == label_id).sum(axis=(1, 2)) > 0]
                bpr_json_per_label[key] = bpr_scores_per_label

            with open(self.output_fnames[bpr_file], "wb") as f:
                pickle.dump(bpr_json_per_label, f)


def crop(task_string, override=False, num_threads=default_num_threads):
    cropped_out_dir = join(clNet_cropped_data, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4d_output_dir_task = join(clNet_raw_data, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(join(clNet_raw_data, task_string, "dataset.json"), cropped_out_dir)


def analyze_dataset(task_string, override=False, collect_intensityproperties=True, num_processes=default_num_threads):
    cropped_out_dir = join(clNet_cropped_data, task_string)
    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=override, num_processes=num_processes)
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)


def add_classes_in_slice_info(args):
    """
    We need this for 2D dataloader with oversampling. As of now it will detect slices that contain specific classes
    at run time, meaning it needs to iterate over an entire patient just to extract one slice. That is obviously bad,
    so we are doing this once beforehand and just give the dataloader the info it needs in the patients pkl file.

    """
    npz_file, pkl_file, all_classes = args
    seg_map = np.load(npz_file)["data"][-1]
    with open(pkl_file, "rb") as f:
        props = pickle.load(f)
    # if props.get("classes_in_slice_per_axis") is not None:
    print(pkl_file)
    # this will be a dict of dict where the first dict encodes the axis along which a slice is extracted in its keys.
    # The second dict (value of first dict) will have all classes as key and as values a list of all slice ids that
    # contain this class
    classes_in_slice = OrderedDict()
    for axis in range(3):
        other_axes = tuple([i for i in range(3) if i != axis])
        classes_in_slice[axis] = OrderedDict()
        for c in all_classes:
            valid_slices = np.where(np.sum(seg_map == c, axis=other_axes) > 0)[0]
            classes_in_slice[axis][c] = valid_slices

    number_of_voxels_per_class = OrderedDict()
    for c in all_classes:
        number_of_voxels_per_class[c] = np.sum(seg_map == c)

    props["classes_in_slice_per_axis"] = classes_in_slice
    props["number_of_voxels_per_class"] = number_of_voxels_per_class

    with open(pkl_file, "wb") as f:
        pickle.dump(props, f)
