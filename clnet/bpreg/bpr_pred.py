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


import os
import sys
import cv2
import json
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.models as models

from clnet.paths import default_bpreg_model
from clnet.bpreg.preprocessing.nifti2npy import Nifti2Npy
from clnet.bpreg.utils.json_parser import parse_json4kaapana
from clnet.bpreg.score_processing import Scores, BodyPartExaminedDict
from clnet.bpreg.network_architecture.base_model import BodyPartRegressionBase
from clnet.bpreg.score_processing.bodypartexamined_tag import BodyPartExaminedTag, BODY_PARTS_INCLUDED, DISTINCT_BODY_PARTS, MIN_PRESENT_LANDMARKS

cv2.setNumThreads(1)
sys.path.append("../../")
BODY_PARTS = {
    "legs": [np.nan, "pelvis_start"],
    "pelvis": ["pelvis_start", "pelvis_end"],
    "abdomen": ["L5", "Th8"],
    "chest": ["Th12", "Th1"],
    "shoulder-neck": ["Th3", "C2"],
    "head": ["C5", np.nan],
}
DEFAULT_MODEL = default_bpreg_model


def bpr_gen(ct_input_filename_list_raw, bpr_output_filename_list_raw, overwrite_existing=False, verbose=True):
    # parse input filenames
    ct_input_filename_list, bpr_output_filename_list = [], []
    for ct_input_filename, bpr_output_filename in zip(ct_input_filename_list_raw, bpr_output_filename_list_raw):
        if not os.path.isfile(bpr_output_filename) or overwrite_existing:
            ct_input_filename_list.append(ct_input_filename)
            bpr_output_filename_list.append(bpr_output_filename)
    # gen bpr
    # bpreg_for_list(ct_input_filename_list, bpr_output_filename_list)
    if len(ct_input_filename_list) == len(bpr_output_filename_list) and len(ct_input_filename_list) > 0:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
            processes = []
            for i in range(num_gpus):
                p = mp.Process(target=bpreg_predict_on_device,
                               args=(ct_input_filename_list[i::num_gpus], bpr_output_filename_list[i::num_gpus], devices[i], verbose))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            bpreg_predict_on_device(ct_input_filename_list, bpr_output_filename_list, "cpu", verbose)


def bpreg_predict_on_device(input_files: list, output_files: list, device, verbose: bool = True, stringify_json: bool = False):
    if len(input_files) > 0:
        if device != "cpu":
            torch.cuda.set_device(device)
        if verbose:
            print("Generating bpr scores on", device, output_files)
        bpr_infer = InferenceBPR(device)
        for input_file, output_file in zip(input_files, output_files):
            try:
                bpr_infer.npy_with_spacing2json(input_file, output_file, stringify_json=stringify_json)
            except:
                bpr_infer.nifti2json(input_file, output_file, stringify_json=stringify_json)
        del bpr_infer
        if device != "cpu":
            torch.cuda.empty_cache()


class InferenceBPR:
    """
    Adapted from Body Part Regression Model for inference purposes.
    """

    def __init__(self, device, base_dir: str = DEFAULT_MODEL, warning_to_error: bool = False):

        self.base_dir = base_dir
        self.device = device
        self.model = load_model(base_dir, device=self.device)
        self.load_inference_settings()
        self.n2n = Nifti2Npy(target_pixel_spacing=3.5, min_hu=-1000, max_hu=1500, size=128)
        self.warning_to_error = warning_to_error

    def load_inference_settings(self):

        path = os.path.join(self.base_dir, "inference-settings.json")
        if not os.path.exists(path):
            print("WARNING: For this model, no inference settings can be load!")

        with open(path, "rb") as f:
            settings = json.load(f)

        # use for inference the lookuptable from all predictions
        # of the annotated landmarks in the train- and validation-dataset
        self.lookuptable_original = settings["lookuptable_train_val"]["original"]
        self.lookuptable = settings["lookuptable_train_val"]["transformed"]

        self.start_landmark = settings["settings"]["start-landmark"]
        self.end_landmark = settings["settings"]["end-landmark"]

        self.transform_min = self.lookuptable_original[self.start_landmark]["mean"]
        self.transform_max = self.lookuptable_original[self.end_landmark]["mean"]

        self.slope_mean = settings["slope_mean"]
        self.tangential_slope_min = settings["lower_quantile_tangential_slope"]
        self.tangential_slope_max = settings["upper_quantile_tangential_slope"]

    def predict_tensor(self, tensor, n_splits=200):
        scores = []
        n = tensor.shape[0]
        slice_splits = list(np.arange(0, n, n_splits))
        slice_splits.append(n)

        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)
            for i in range(len(slice_splits) - 1):
                min_index = slice_splits[i]
                max_index = slice_splits[i + 1]
                score = self.model(tensor[min_index:max_index, :, :, :].to(self.device))
                scores += [s.item() for s in score]

        scores = np.array(scores)
        return scores

    def predict_nifti(self, nifti_path: str):
        # get nifti file as tensor
        try:
            x, pixel_spacings = self.n2n.preprocess_nifti(nifti_path)
        except:
            x, pixel_spacings = np.nan, np.nan

        if isinstance(x, float) and np.isnan(x):
            x, pixel_spacings = self.n2n.load_volume(nifti_path)
            if not isinstance(x, np.ndarray):
                if self.warning_to_error:
                    raise ValueError(f"File {nifti_path} can not be loaded.")
                return np.nan

            warning_msg = (
                f"File {nifti_path.split('/')[-1]} with shape {x.shape} and pixel spacings "
                f"{pixel_spacings} can not be converted to a 3-dimensional volume of the size {self.n2n.size}x{self.n2n.size}xz;"
            )
            print("WARNING: ", warning_msg)
            if self.warning_to_error:
                raise ValueError(warning_msg)
            return np.nan

        x = np.transpose(x, (2, 0, 1))[:, np.newaxis, :, :]
        x_tensor = torch.tensor(x)
        x_tensor.to(self.device)

        # predict slice-scores
        scores = self.predict_tensor(x_tensor)
        return self.parse_scores(scores, pixel_spacings[2])

    def predict_npy_with_spacing(self, npy_path: str):
        # get nifti file as tensor
        try:
            x, pixel_spacings = self.n2n.preprocess_npy_with_spacing(npy_path)
        except:
            x, pixel_spacings = np.nan, np.nan

        if isinstance(x, float) and np.isnan(x):
            x = self.n2n.preprocess_npy(npy_path, pixel_spacings)
            if not isinstance(x, np.ndarray):
                if self.warning_to_error:
                    raise ValueError(f"File {npy_path} can not be loaded.")
                return np.nan

            warning_msg = (
                f"File {npy_path.split('/')[-1]} with shape {x.shape} and pixel spacings "
                f"{pixel_spacings} can not be converted to a 3-dimensional volume of the size {self.n2n.size}x{self.n2n.size}xz;"
            )
            print("WARNING: ", warning_msg)
            if self.warning_to_error:
                raise ValueError(warning_msg)
            return np.nan

        x = np.transpose(x, (2, 0, 1))[:, np.newaxis, :, :]
        x_tensor = torch.tensor(x)
        x_tensor.to(self.device)

        # predict slice-scores
        scores = self.predict_tensor(x_tensor)
        return self.parse_scores(scores, pixel_spacings[2])

    def parse_scores(self, scores_array, pixel_spacing):

        scores = Scores(
            scores_array,
            pixel_spacing,
            transform_min=self.lookuptable_original[self.start_landmark]["mean"],
            transform_max=self.lookuptable_original[self.end_landmark]["mean"],
            slope_mean=self.slope_mean,
            tangential_slope_min=self.tangential_slope_min,
            tangential_slope_max=self.tangential_slope_max,
        )
        return scores

    def nifti2json(self, nifti_path: str, output_path: str = "", stringify_json: bool = False, ignore_invalid_z: bool = False):
        """
        Main method to convert NIFTI CT volumes int JSON meta data files.
        Args:
            nifti_path (str): path of input NIFTI file
            output_path (str): output path to save JSON file
            stringify_json (bool): Set it to true for Kaapana JSON format
            axis_ordering (tuple): Axis ordering of CT volume. (0,1,2) is equivalent to the axis ordering xyz.
            ignore_invalid_z (bool): If true, than invalid z-spacing will be ignored for predicting the body part examined and not NONE will be given back.
        """
        slice_scores = self.predict_nifti(nifti_path)
        if isinstance(slice_scores, float) and np.isnan(slice_scores):
            return np.nan

        data_storage = VolumeStorage(slice_scores, self.lookuptable, ignore_invalid_z=ignore_invalid_z)
        if len(output_path) > 0:
            data_storage.save_json(output_path, stringify_json=stringify_json)
        return data_storage.json

    def npy_with_spacing2json(self, npy_path: str, output_path: str = "", stringify_json: bool = False, ignore_invalid_z: bool = False):
        slice_scores = self.predict_npy_with_spacing(npy_path)
        if isinstance(slice_scores, float) and np.isnan(slice_scores):
            return np.nan

        data_storage = VolumeStorage(slice_scores, self.lookuptable, ignore_invalid_z=ignore_invalid_z)
        if len(output_path) > 0:
            data_storage.save_json(output_path, stringify_json=stringify_json)
        return data_storage.json


@dataclass
class VolumeStorage:
    """Body part metadata for one volume

    Args:
        scores (Scores): predicted slice scores
        lookuptable (dict): reference table which contains expected scores for anatomies
        body_parts ([type], optional): dictionary to define the body parts for the tag: "body part examined". Defaults to BODY_PARTS.
        body_parts_included ([type], optional): dictionary to calculate the "body part examined tag". Defaults to BODY_PARTS_INCLUDED.
        distinct_body_parts ([type], optional): dictionary to calculate the "body part examined tag". Defaults to DISTINCT_BODY_PARTS.
        min_present_landmarks ([type], optional): dictionary to calculate the "body part examined rtag". Defaults to MIN_PRESENT_LANDMARKS.
    """

    def __init__(
            self,
            scores: Scores,
            lookuptable: dict,
            body_parts=BODY_PARTS,
            body_parts_included=BODY_PARTS_INCLUDED,
            distinct_body_parts=DISTINCT_BODY_PARTS,
            min_present_landmarks=MIN_PRESENT_LANDMARKS,
            ignore_invalid_z: bool = False,
    ):
        self.ignore_invalid_z = ignore_invalid_z
        self.body_parts = body_parts
        self.body_parts_included = body_parts_included
        self.distinct_body_parts = distinct_body_parts
        self.min_present_landmarks = min_present_landmarks

        self.cleaned_slice_scores = list(scores.values.astype(np.float64))
        self.z = list(scores.z.astype(np.float64))
        self.unprocessed_slice_scores = list(
            scores.original_transformed_values.astype(np.float64)
        )
        self.lookuptable = lookuptable

        self.zspacing = float(scores.zspacing)  # .astype(np.float64)
        self.reverse_zordering = float(scores.reverse_zordering)
        self.valid_zspacing = float(scores.valid_zspacing)
        self.expected_slope = float(scores.slope_mean)
        self.observed_slope = float(scores.a)
        self.expected_zspacing = float(scores.expected_zspacing)
        self.r_slope = float(scores.r_slope)
        self.bpe = BodyPartExaminedDict(lookuptable, body_parts=self.body_parts)
        self.bpet = BodyPartExaminedTag(
            lookuptable,
            body_parts_included=self.body_parts_included,
            distinct_body_parts=self.distinct_body_parts,
            min_present_landmarks=self.min_present_landmarks,
            ignore_invalid_z=self.ignore_invalid_z,
        )

        self.settings = {
            "slice score processing": scores.settings,
            "body part examined dict": self.body_parts,
            "body part examined tag": {
                "body parts included": self.body_parts_included,
                "distinct body parts": self.distinct_body_parts,
                "min present landmarks": self.min_present_landmarks,
            },
        }

        self.json = {
            "cleaned slice scores": self.cleaned_slice_scores,
            "z": self.z,
            "unprocessed slice scores": self.unprocessed_slice_scores,
            "body part examined": self.bpe.get_examined_body_part(
                self.cleaned_slice_scores
            ),
            "body part examined tag": self.bpet.estimate_tag(scores),
            "look-up table": self.lookuptable,
            "reverse z-ordering": self.reverse_zordering,
            "valid z-spacing": self.valid_zspacing,
            "expected slope": self.expected_slope,
            "observed slope": self.observed_slope,
            "slope ratio": self.r_slope,
            "expected z-spacing": self.expected_zspacing,
            "z-spacing": self.zspacing,
            "settings": self.settings,
        }

    def save_json(self, output_path: str, stringify_json=False):
        """Store data in json file

        Args:
            output_path (str): save path for json file
            stringify_json (bool, optional): if True, stringify output of parameters and
            convert json file to a Kaapana friendly format
        """
        data = self.json
        if stringify_json:
            data = parse_json4kaapana(data)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)


def load_model(base_dir, model_file="model.pt", device="cuda"):
    model_filepath = os.path.join(base_dir, model_file)

    model = BodyPartRegression().to(device)
    model.load_state_dict(torch.load(model_filepath, map_location=torch.device(device), weights_only=False), strict=False)
    model.eval()
    model.to(device)

    return model


class BodyPartRegression(BodyPartRegressionBase):
    def __init__(self, lr: float = 1e-4, lambda_: float = 0, alpha: float = 0, pretrained: bool = False, delta_z_max: float = np.inf,
                 loss_order: str = "h", beta_h: float = 0.025, alpha_h: float = 0.5, weight_decay: int = 0):
        BodyPartRegressionBase.__init__(self, lr=lr, lambda_=lambda_, alpha=alpha, pretrained=pretrained, delta_z_max=delta_z_max,
                                        loss_order=loss_order, beta_h=beta_h, alpha_h=alpha_h, weight_decay=weight_decay)
        # load vgg base model
        self.conv6 = nn.Conv2d(512, 512, 1, stride=1, padding=0)  # in_channel, out_channel, kernel_size
        self.fc7 = nn.Linear(512, 1)
        self.model = self.get_vgg()

    def get_vgg(self):
        vgg = models.vgg16(weights=None)
        vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        return vgg.features

    def forward(self, x: torch.Tensor):
        x = self.model(x.float())
        x = F.relu(self.conv6(x))
        x = torch.mean(x, dim=(2, 3))
        x = x.view(-1, 512)
        x = self.fc7(x)
        return x
