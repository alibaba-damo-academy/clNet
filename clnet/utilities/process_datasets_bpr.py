# -*- coding: utf-8 -*-
"""
Created on 7/14/2023
Author: Puyang Wang
"""

import torch
from bpreg.settings import *
from bpreg.inference.inference_model import InferenceModel

def bpreg_for_list(
    ifiles: list,
    ofiles: list,
    skip_existing: bool = False,
    stringify_json: bool = False,
    gpu_available: bool = True,
):
    model_path = DEFAULT_MODEL
    # test if gpu is available
    if not torch.cuda.is_available():
        gpu_available = False
    model = InferenceModel(model_path, gpu=gpu_available)

    for ifile, ofile in zip(ifiles, ofiles):

        if os.path.exists(ofile) and skip_existing:
            # print(f"JSON-file already exists. Skip file: {ifile}")
            continue

        # print(f"Create body-part meta data file: {ofile}")
        model.nifti2json(ifile, ofile, stringify_json=stringify_json)
    del model
    torch.cuda.empty_cache()
    return None