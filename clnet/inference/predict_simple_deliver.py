#   Author @Dazhou Guo
#   Data: 08.20.2023

import time
import torch.multiprocessing as mp

from clnet.configuration import *
from clnet.paths import default_plans_identifier
from clnet.preprocessing.bm_bpr_gen import bm_bpr_gen
from clnet.inference.utils import *
from clnet.inference.load_decoder_and_predict_on_device import load_model_and_predict
from clnet.inference.cfg_parser import cfg_parser_for_inference


def clnet_inference(clnet_cfg: str, input_folder: str, output_folder: str, task_plan_to_use: str = None, num_threads_preprocessing: int = default_num_threads,
                    disable_tta: bool = True, expected_num_modalities: int = 1, num_threads_nifti_save: int = default_num_threads,
                    overwrite_existing_preprocessing: bool = False, fast: bool = False, overwrite_existing_pred: bool = False, compile_network: bool = True,
                    save_intermediate_result_for_debug: bool = False, all_in_gpu: bool = False, step_size: float = 0.5, disable_mixed_precision: bool = False,
                    do_not_inference: bool = False, disable_bm_bpr: bool = False):
    start_t = time.time()
    # set multiprocessing start method -> spawn
    mp.set_start_method("spawn", force=True)
    # preprocessing
    maybe_mkdir_p(output_folder)
    case_ids = check_input_folder_and_return_case_ids(input_folder, expected_num_modalities)
    list_of_lists, bpr_files, bm_files = bm_bpr_gen(input_folder, output_folder, overwrite_existing_preprocessing, num_threads_preprocessing, disable_bm_bpr)

    # parse the input cfg json file for inference
    trainer_class, trainer_heads_summarized, trainer_heads_details = cfg_parser_for_inference(clnet_cfg, default_plans_identifier, task_plan_to_use)
    end_t = time.time()
    print("Preprocessing Time: %f" % (end_t - start_t))
    start_t = time.time()

    if fast:
        # TODO: faster resampling and faster saving to nii
        step_size = 0.8
        all_in_gpu = True
        disable_tta = True

    # Inference -- decoders
    load_model_and_predict(trainer_class, trainer_heads_summarized, trainer_heads_details, "decoders",
                           task_plan_to_use, do_not_inference, case_ids, list_of_lists, bpr_files, bm_files,
                           output_folder, num_threads_preprocessing, num_threads_nifti_save,
                           disable_tta, overwrite_existing_pred, compile_network, save_intermediate_result_for_debug,
                           all_in_gpu, disable_mixed_precision, step_size)
    # Inference -- supporting
    load_model_and_predict(trainer_class, trainer_heads_summarized, trainer_heads_details, "supporting",
                           task_plan_to_use, do_not_inference, case_ids, list_of_lists, bpr_files, bm_files,
                           output_folder, num_threads_preprocessing, num_threads_nifti_save,
                           disable_tta, overwrite_existing_pred, compile_network, save_intermediate_result_for_debug,
                           all_in_gpu, disable_mixed_precision, step_size)
    end_t = time.time()
    print("Inference Time: %f" % (end_t - start_t))


if __name__ == "__main__":

    clnet_cfg_current = "/mnt/nas/suyanzhou.syz/repo/clNet_inference/clnet/training_cfg_json/CSS_GeneralEncoder_ft_for_inference.json"
    input_folder_current = "/mnt/nas/suyanzhou.syz/dataset/Totalseg_test_example/imagesVal"
    output_folder_current = "/mnt/nas/suyanzhou.syz/dataset/Totalseg_test_example/imagesVal_pred"

    clnet_inference(clnet_cfg_current, input_folder_current, output_folder_current, fast=False,
                    overwrite_existing_preprocessing=False, overwrite_existing_pred=True,
                    save_intermediate_result_for_debug=True, all_in_gpu=True, compile_network=True, disable_bm_bpr=False)
