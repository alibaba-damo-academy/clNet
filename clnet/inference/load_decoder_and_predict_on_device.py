#   Author @Dazhou Guo
#   Data: 08.15.2024

import time
from typing import Optional

import torch.cuda
import torch.multiprocessing as mp

from clnet.inference.utils import *
from clnet.inference.cfg_parser import cfg_parser_for_device
from clnet.postprocessing.resample_seg import postprocess_resample_seg
from clnet.network_architecture.custom_modules.pruning_modules import *

from batchgenerators.utilities.file_and_folder_operations import *


def load_model_and_predict(trainer_class: Optional, trainer_heads_summarized: dict, trainer_heads_details: dict,
                           decoder_or_support: str, task_plan_to_use: str, do_not_inference: bool, case_ids: np.ndarray,
                           list_of_lists: list, bpr_files: list, bm_files: list, output_folder: str,
                           num_threads_preprocessing: int, num_threads_nifti_save: int, disable_tta: bool,
                           overwrite_existing_pred: bool, compile_network: bool, save_intermediate_result_for_debug: bool, all_in_gpu: bool,
                           disable_mixed_precision: bool, step_size: float):
    # Try to parse all decoder/supporting heads
    all_decoders = list(trainer_heads_summarized[decoder_or_support].keys())
    if len(all_decoders) > 0:
        task_dict_for_device, heads_to_pred_cleaned = cfg_parser_for_device(trainer_heads_details, trainer_heads_summarized, all_decoders, decoder_or_support)
        if torch.cuda.is_available():
            num_gpus = min(torch.cuda.device_count(), len(heads_to_pred_cleaned))
            # num_gpus = 1  # for debug only
            devices = [torch.device(f'cuda:{n}') for n in range(num_gpus)]
            # If the number of GPUs or heads is less than 2, we do not need to use multiprocessing
            if num_gpus < 2:
                load_model_and_predict_on_device(
                    devices[0], trainer_class, task_dict_for_device, trainer_heads_summarized, heads_to_pred_cleaned, decoder_or_support, task_plan_to_use,
                    do_not_inference, case_ids, list_of_lists, bpr_files, bm_files, output_folder, num_threads_preprocessing,
                    disable_tta, overwrite_existing_pred, compile_network, save_intermediate_result_for_debug, all_in_gpu, disable_mixed_precision, step_size)
            else:
                processes = []
                for n in range(num_gpus):
                    if len(heads_to_pred_cleaned) == 0:
                        continue
                    heads_to_pred = heads_to_pred_cleaned[n::num_gpus]
                    p = mp.Process(target=load_model_and_predict_on_device,
                                   args=(devices[n], trainer_class, task_dict_for_device, trainer_heads_summarized, heads_to_pred, decoder_or_support,
                                         task_plan_to_use, do_not_inference, case_ids, list_of_lists, bpr_files, bm_files, output_folder,
                                         num_threads_preprocessing, disable_tta, overwrite_existing_pred, compile_network, save_intermediate_result_for_debug,
                                         all_in_gpu, disable_mixed_precision, step_size))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
        else:
            if len(heads_to_pred_cleaned) > 0:
                load_model_and_predict_on_device(
                    "cpu", trainer_class, task_dict_for_device, trainer_heads_summarized, heads_to_pred_cleaned, decoder_or_support, task_plan_to_use,
                    do_not_inference, case_ids, list_of_lists, bpr_files, bm_files, output_folder, num_threads_preprocessing,
                    disable_tta, overwrite_existing_pred, compile_network, save_intermediate_result_for_debug, all_in_gpu, disable_mixed_precision, step_size)
        print("Resampling and Saving Predictions...")
        postprocess_resample_seg(trainer_heads_summarized, output_folder, case_ids, decoder_or_support,
                                 overwrite_existing_pred, num_threads_nifti_save, save_intermediate_result_for_debug)


def load_model_and_predict_on_device(device: str, trainer_class: Optional, trainer_heads_details: dict, trainer_heads_summarized: dict,
                                     heads_to_pred: list, decoder_or_support: str, task_plan_to_use: str, do_not_inference: bool, case_ids: np.ndarray,
                                     list_of_lists: list, bpr_files: list, bm_files: list, output_folder: str,
                                     num_threads_preprocessing: int, disable_tta: bool,  overwrite_existing_pred: bool, compile_network: bool,
                                     save_intermediate_result_for_debug: bool, all_in_gpu: bool, disable_mixed_precision: bool, step_size: float):
    pretrained_network_continual_learning = None
    trainer = None
    if device != "cpu":
        torch.cuda.set_device(device)
    # try to load model(s) using cfg
    if not do_not_inference:
        task_plans = {}
        task_stages = {}
        # Load all pre-trained models
        ge_plans = None
        ge_stage = None
        train_dict_decoders = {}
        train_dict_supporting = {}
        start_t = time.time()
        # load model one task at a time -- all in CPU/MEM
        for n, task in enumerate(trainer_heads_details):
            plans_file = trainer_heads_details[task]["plans_file"]
            fold = trainer_heads_details[task]["fold"]
            output_folder_name = trainer_heads_details[task]["output_folder_name"]
            stage = trainer_heads_details[task]["stage"]
            trainer = trainer_class(trainer_heads_details, task, "load_all", "all", pretrained_network_continual_learning,
                                    plans_file, fold, output_folder=output_folder_name, stage=stage)
            # We store all plans from every tasks
            trainer.load_plans_file()
            trainer.process_plans(trainer.plans)
            trainer.setup_data_aug_params()
            task_plans[task] = trainer.plans
            task_stages[task] = trainer.stage
            train_dict_decoders.update(trainer.train_dict["decoders"])
            train_dict_supporting.update(trainer.train_dict["supporting"])
            if n == 0:
                ge_plans = trainer.plans
                ge_stage = trainer.stage
            trainer.initialize_network()  # all task should call initialize_network to load trained model properly
            trainer.load_pretrained_params_ensemble(load_from_ema=True, is_training=False)
            pretrained_network_continual_learning = trainer.network.to(torch.device("cpu"))

        del pretrained_network_continual_learning
        if task_plan_to_use is None or len(task_plan_to_use) == 0 or task_plan_to_use not in task_plans:
            trainer.plans = ge_plans
            trainer.stage = ge_stage
        else:
            trainer.plans = task_plans[task_plan_to_use]
            trainer.stage = task_stages[task_plan_to_use]
        #  After re-assign the "task_plans" to the "trainer", we re-initialize the plan again
        trainer.train_dict["decoders"] = train_dict_decoders
        trainer.train_dict["supporting"] = train_dict_supporting
        trainer.process_plans(trainer.plans)
        trainer.setup_data_aug_params()
        # After loading all the weights -> remove the pruning params and make all pruned weights to be 0.
        remove_pruning_reparam(trainer.network.decoder_dict)
        # decoder_sparsity = measure_network_sparsity(trainer.network.decoder_dict)
        # if decoder_sparsity is not None:
        #     trainer.print_to_log_file("Decoder Sparsity: %s" % decoder_sparsity)
        # Remove all EMA modules
        trainer.network.ema_dict.clear()
        # After loading all pruned weights, we copy the model to GPU for inference.
        if torch.cuda.is_available():
            trainer.network.to(device)
        end_t = time.time()
        trainer.print_to_log_file("Model Loading Time on Device: %s -- %.1f (s)" % (device, end_t - start_t))
        if compile_network:
            # compile will take an extra over-head time (seconds), e.g., 10s on P100 GPU, 8s on V100 GPU.
            trainer.compile_network(False)
        for head_to_pred in heads_to_pred:
            predict_cases(trainer, case_ids, list_of_lists, trainer_heads_summarized, decoder_or_support,
                          bpr_files, bm_files, output_folder, head_to_pred, num_threads_preprocessing, not disable_tta,
                          overwrite_existing=overwrite_existing_pred, save_intermediate_result_for_debug=save_intermediate_result_for_debug,
                          all_in_gpu=all_in_gpu, mixed_precision=not disable_mixed_precision, step_size=step_size)


def predict_cases(trainer: Optional, case_ids: np.ndarray, list_of_lists: list, trainer_heads_summarized: dict, decoder_or_support: str,
                  bpr_files: list, bm_files: list, output_folder: str, head_to_pred: str, num_threads_preprocessing: int, do_tta: bool,
                  overwrite_existing: bool = True, save_intermediate_result_for_debug: bool = False, all_in_gpu: bool = None,
                  mixed_precision: bool = True, step_size: float = 0.5):
    # check if perform all processing in GPU
    trainer.print_to_log_file("'{}' to Infer: {}".format(decoder_or_support.upper(), head_to_pred))

    trainer.plans_file = trainer.task_dict[trainer_heads_summarized[decoder_or_support][head_to_pred]["task"]]["plans_file"]
    trainer.load_plans_file()
    trainer.process_plans(trainer.plans)
    trainer.setup_data_aug_params()
    trainer.get_plan_properties()

    patch_size = trainer.patch_size
    if trainer_heads_summarized["patch_size"][head_to_pred] is not None:
        patch_size = trainer_heads_summarized["patch_size"][head_to_pred]

    # trainer.get_intensity_properties()
    if all_in_gpu is None:
        all_in_gpu = False
    else:
        all_in_gpu = bool(all_in_gpu)

    # output path setup
    maybe_mkdir_p(output_folder)
    output_filenames_nii = [join(output_folder, i, i + "_{}_{}.nii.gz".format(head_to_pred, decoder_or_support)) for i in case_ids]
    output_filenames_npy = [join(output_folder, i, i + "_{}_{}.npy".format(head_to_pred, decoder_or_support)) for i in case_ids]
    header_json_filenames = [join(output_folder, i, i + "_{}_{}.json".format(head_to_pred, decoder_or_support)) for i in case_ids]
    assert len(list_of_lists) == len(output_filenames_nii) == len(output_filenames_npy)

    # check existing predictions
    cleaned_output_files = []
    for o in output_filenames_nii:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))
    # if "overwrite_existing" is false, then we will remove existing predictions from inference list and keep predicted mask.
    if not overwrite_existing:
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not isfile(j))]
        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        bpr_files = [bpr_files[i] for i in not_done_idx]
        bm_files = [bm_files[i] for i in not_done_idx]
        header_json_filenames = [header_json_filenames[i] for i in not_done_idx]

    # load the body part scores and pre-set patch-size for each decoding head
    model_bpr_lookup = trainer_heads_summarized['bpr_range']
    # find the slice range for head_to_infer according to bpr scores and model_bpr_lookup,
    # if the bpr range of head_to_infer is not overlapped with bpr scores of input CT,
    # then case will be removed from inference list and outputs an empty mask.
    if model_bpr_lookup is not None:
        head_to_infer_brp_range_for_all_images = []
        bpr = model_bpr_lookup
        bottom_bpr = float("inf")
        top_bpr = float("-inf")
        if head_to_pred != 'all':
            if "gtv" in head_to_pred.lower() or "tumor" in head_to_pred.lower():
                bottom_bpr = min(bpr[head_to_pred]['min'] - bpr[head_to_pred]['std'], bottom_bpr)
                top_bpr = max(bpr[head_to_pred]['max'] + bpr[head_to_pred]['std'], top_bpr)
            else:
                bottom_bpr = min(bpr[head_to_pred]['percentile_00_5'] - bpr[head_to_pred]['std'], bottom_bpr)
                top_bpr = max(bpr[head_to_pred]['percentile_99_5'] + bpr[head_to_pred]['std'], top_bpr)

        # check if the input image is out of the bpr range
        excluded_idx = []
        if bpr_files is not None and isinstance(bpr_files, list):
            for i, bpr_file in enumerate(bpr_files):
                with open(bpr_file) as f:
                    bpr_single_case = json.load(f)
                scores = np.array(bpr_single_case["cleaned slice scores"])
                slices = np.where(np.logical_and(scores >= bottom_bpr, scores <= top_bpr))
                if len(slices[0]) == 0:
                    print("Warning! {} does not has any anatomical structure in {}, skipping inference and exporting empty mask.".
                          format(os.path.basename(list_of_lists[i][0]), head_to_pred))
                    excluded_idx.append(i)
                    # creating empty masks
                    bm = sitk.ReadImage(bm_files[i])
                    pred_data = np.zeros_like(sitk.GetArrayFromImage(bm))
                    np.save(output_filenames_npy[i], np.array(pred_data, dtype=np.uint16))
                else:
                    bottom_slice = int(np.min(slices))
                    top_slice = int(np.max(slices)) + 1
                    head_to_infer_brp_range_for_all_images.append([bottom_slice, top_slice])
        # removing cases that has no overlapped bpr ranges.
        list_of_lists = [ele for idx, ele in enumerate(list_of_lists) if idx not in excluded_idx]
        cleaned_output_files = [ele for idx, ele in enumerate(cleaned_output_files) if idx not in excluded_idx]
        if bm_files is not None:
            bm_files = [ele for idx, ele in enumerate(bm_files) if idx not in excluded_idx]
        header_json_filenames = [ele for idx, ele in enumerate(header_json_filenames) if idx not in excluded_idx]
        output_filenames_npy = [ele for idx, ele in enumerate(output_filenames_npy) if idx not in excluded_idx]
    else:
        head_to_infer_brp_range_for_all_images = None

    # initializing multithread preprocessing
    preprocessing = preprocess_multithreaded(trainer.plans, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             header_json_filenames=header_json_filenames, output_filenames_npy=output_filenames_npy,
                                             bm_files=bm_files, bpr_range=head_to_infer_brp_range_for_all_images)
    # specify the head to predict
    if head_to_pred is None:
        head_to_pred = 'all'
    if decoder_or_support is None:
        decoder_or_support = trainer.decoder_or_support
    # clear GPU memory & start inference
    torch.cuda.empty_cache()
    for preprocessed in preprocessing:
        output_filename, (data_preprocessed, properties, header_json_filename, output_filename_npy) = preprocessed
        if isinstance(data_preprocessed, str):
            data = np.load(data_preprocessed)
            os.remove(data_preprocessed)
            data_preprocessed = data

        trainer.print_to_log_file("predicting", output_filename)
        res_cropped = trainer.predict_preprocessed_data_return_seg_and_softmax_ensemble(
            data_preprocessed, decoder_or_support, head_to_pred, do_mirroring=do_tta,
            mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision, current_patch_size=patch_size, verbose=False, is_inference=True)

        dump_to_json(properties, header_json_filename)
        np.save(output_filename_npy, np.array(res_cropped[0]))
        if save_intermediate_result_for_debug:
            np.save(output_filename_npy[:-len(".npy")] + "_one_hot.npy", np.array(res_cropped[1]))
        del res_cropped


def dump_to_json(properties, header_json_filename):
    for key in properties:
        if isinstance(properties[key], np.ndarray):
            properties[key] = list(properties[key])
        if isinstance(properties[key], list):
            for j in range(len(properties[key])):
                if isinstance(properties[key][j], np.integer):
                    properties[key][j] = int(properties[key][j])
                elif isinstance(properties[key][j], np.floating):
                    properties[key][j] = float(properties[key][j])
    with open(header_json_filename, "w") as f:
        json.dump(properties, f)
