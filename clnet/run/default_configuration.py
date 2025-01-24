#   Author @Dazhou Guo
#   Data: 03.01.2023
import copy
import shutil
import os.path
import warnings
import numpy as np
from collections import OrderedDict

import clnet
from clnet.configuration import *
from clnet.training.model_restore import recursive_find_python_class
from clnet.utilities.task_name_id_conversion import convert_id_to_task_name
from clnet.paths import preprocessing_output_dir, default_plans_identifier
from clnet.run.config_utils import *


def get_continual_decoding_ensemble_setup(pth_json, plans_identifier=default_plans_identifier, load_plan_from_pretrain=False,
                                          rank=0, is_ddp=False, inference=False):
    if not os.path.exists(pth_json):
        txt = "Cannot find the training JSON file: " + pth_json
        raise RuntimeError(txt)
    try:
        dict_setup = load_json(pth_json)
    except:
        dict_setup = {}

    if len(dict_setup) == 0:
        RuntimeError("Cannot open training JSON file. Please double check the training JSON file")

    dict_ret = {}
    if "clnet_network" in dict_setup:
        network = dict_setup["clnet_network"]
        assert network in ["2d", "3d_fullres"], "-m must be 2d or 3d_fullres"
    else:
        network = default_network
    if "clnet_trainer" in dict_setup:
        trainer = dict_setup["clnet_trainer"]
    else:
        trainer = default_trainer

    if is_ddp:
        if not trainer.endswith("DDP"):
            trainer += "_DDP"
    elif "clnet_enable_ddp" in dict_setup and dict_setup["clnet_enable_ddp"]:
        if not trainer.endswith("DDP"):
            trainer += "_DDP"

    if inference:
        if trainer.endswith("DDP"):
            trainer = trainer[:-len("_DDP")]
    task_name_ge = ""
    if "clnet_general_encoder" in dict_setup:
        task_name_ge = dict_setup["clnet_general_encoder"]

    search_in = (clnet.__path__[0], "training", "network_training")
    base_module = 'clnet.training.network_training'
    trainer_class = recursive_find_python_class([join(*search_in)], trainer, current_module=base_module)
    print_with_rank("My trainer class is: ", trainer_class, rank=rank)

    dict_training_order = {}
    flag_no_ge = True
    duplicate_ge = []
    if "clnet_network" in dict_setup:
        del dict_setup["clnet_network"]
    if "clnet_trainer" in dict_setup:
        del dict_setup["clnet_trainer"]
    if "clnet_general_encoder" in dict_setup:
        del dict_setup["clnet_general_encoder"]
    if "clnet_enable_ddp" in dict_setup:
        del dict_setup["clnet_enable_ddp"]

    # ################  Get the General Encoder Setup ################
    # We first check if the "general encoder" is defined in "clnet_general_encoder" tag.
    # If "clnet_general_encoder" is defined, we will use this task to parse general encoder,
    # e.g., load GE from this task, use encoder architecture from this task
    if task_name_ge in dict_setup:
        flag_no_ge = False
        check_ge(task_name_ge, dict_setup, dict_ret)
        del dict_setup[task_name_ge]
    # Then, we check if there exists "general encoder" in the rest of the config JSON
    for task in dict_setup:
        # first, check if there exists 'general encoder'
        # Keys are: "general" + "encoder"
        condition_1 = "type" in dict_setup[task] and "general" in dict_setup[task]["type"].lower() and "encoder" in dict_setup[task]["type"].lower()
        condition_2 = "general" in task.lower() and "encoder" in task.lower()
        if isinstance(dict_setup[task], dict):
            if (condition_1 or condition_2) and flag_no_ge:
                flag_no_ge = False
                check_ge(task, dict_setup, dict_ret)
                duplicate_ge.append(task)

    # The JSON must have one GeneralEncoder
    if flag_no_ge:
        raise RuntimeError("General Encoder is not defined!")
    # The JSON only has one GeneralEncoder, and set the other "General Encoders" as downstream tasks
    if len(duplicate_ge) > 1:
        warning_msg = "Duplicate general encoders are defined %s" % duplicate_ge
        print(warning_msg)
        # # raise RuntimeError("Duplicate general encoders are defined %s" % duplicate_ge)
        # for task in duplicate_ge[1:]:
        #     del dict_setup[task]
    # Delete the GeneralEncoder in "dict_setup" after parsing
    if len(duplicate_ge) > 0:
        del dict_setup[duplicate_ge[0]]

    if len(dict_setup) > 0:
        # ################  Get the EXISTING pre-defined training orders from the JSON
        for task in dict_setup:
            if "train_order" in dict_setup[task]:
                # making sure that GeneralEncoder is ordered 0 is the smallest, and no repeated order.
                current_order = abs(int(dict_setup[task]["train_order"])) + 1
                dict_setup[task]["train_order"] = current_order
                if current_order not in dict_training_order:
                    dict_training_order[current_order] = [task]
                else:
                    dict_training_order[current_order].append(task)
        # ################  Re-order the duplicated training tasks
        if len(dict_training_order) > 0:
            max_training_order_provided = max(dict_training_order.keys())
        else:
            max_training_order_provided = 0
        for current_order in dict_training_order:
            if len(dict_training_order[current_order]) > 1:
                print_with_rank("Existing the same order --", current_order, dict_training_order[current_order], rank=rank)
                for task in dict_training_order[current_order][1:]:
                    dict_setup[task]["train_order"] = max_training_order_provided + 1
                    max_training_order_provided += 1
        # ################  Get the NOT PRE-DEFINED training tasks
        for task in dict_setup:
            # we make sure that we are parsing the "tasks: dict"
            if "clnet_general_encoder" not in task and isinstance(dict_setup[task], dict):
                dict_ret[task] = dict_setup[task]
                if "type" in dict_ret[task]:
                    if "tumor" in dict_ret[task]["type"].lower() or "gtv" in dict_ret[task]["type"].lower() or \
                            "abnormal" in dict_ret[task]["type"].lower() or "anomaly" in dict_ret[task]["type"].lower():
                        dict_ret[task]["type"] = "AnomalyEnsemble"
                    else:
                        dict_ret[task]["type"] = "DecodingEnsemble"
                else:
                    dict_ret[task]["type"] = "DecodingEnsemble"
                if "train_order" not in dict_ret[task]:
                    dict_ret[task]["train_order"] = max_training_order_provided + 1
                    max_training_order_provided += 1
                dict_ret[task] = dict_attribute_check(dict_ret[task], task)
                dict_ret[task] = get_model_training_setup(dict_ret[task])
        dict_ret = OrderedDict(sorted(dict_ret.items(), key=lambda x: x[1]["train_order"]))
        # # Remove the duplicated tasks that share the same task name.
        # dict_ret = dict_duplicate_check(dict_ret)

    # ################  Parse the input arguments
    for task in dict_ret.keys():
        if rank == 0:
            print_with_rank("Task -- %s\n###############################################" % task, rank=rank)
        # changes in "current_dict" will automatically be updated in "dict_ret"
        current_dict = dict_ret[task]
        current_task_id = current_dict["task"]
        if not current_task_id.startswith("Task"):
            current_task_id = int(current_task_id)
            current_task_id = convert_id_to_task_name(current_task_id)
        if isinstance(current_dict["fold"], str) and current_dict["fold"] == "all":
            pass
        else:
            current_dict["fold"] = int(current_dict["fold"])
        # If the task does not exist, then raise ERRORs.
        current_plans_file, current_output_folder_name, current_dataset_directory, current_batch_dice, current_stage = \
            get_default_configuration(network, current_task_id, trainer, plans_identifier)
        current_dict["plans_file"] = current_plans_file
        current_dict["output_folder_name"] = current_output_folder_name
        current_dict["dataset_directory"] = current_dataset_directory
        current_dict["batch_dice"] = current_batch_dice
        current_dict["stage"] = current_stage

        # Here we check if we need to load the "plans" from the pretrained model
        try_to_load_from_pretrained_model = \
            current_dict["continue_training"] or current_dict["model_training_setup"]["decoders"] is None or load_plan_from_pretrain
        # Check if to load the "plan" for the "GeneralEncoder"
        if "load_only_encoder" in current_dict:
            try_to_load_from_pretrained_model = try_to_load_from_pretrained_model or current_dict["load_only_encoder"]

        pretrained_plans_file = os.path.join(current_output_folder_name, "plans.json")
        if os.path.exists(pretrained_plans_file) and try_to_load_from_pretrained_model:
            current_dict["plans_file"] = pretrained_plans_file

    # checking decoding heads
    dict_check_decoders = {}
    for task in dict_ret.keys():
        for decoder in dict_ret[task]["decoders"]:
            if decoder not in dict_check_decoders:
                dict_check_decoders[decoder] = \
                    [task, dict_ret[task]["train_order"], dict_ret[task]["decoders"][decoder]]
            else:
                if not isinstance(dict_check_decoders[decoder][-1], type(dict_ret[task]["decoders"][decoder])):
                    error_msg = "The ---'%s'--- output mis-match current 'Task %s-%s' vs. existing 'Task %s-%s'" \
                                % (decoder, task, dict_ret[task]["decoders"][decoder],
                                   dict_check_decoders[decoder][0], dict_check_decoders[decoder][-1])
                    raise RuntimeError(error_msg)
                else:
                    if isinstance(dict_check_decoders[decoder][-1], list):
                        if len(dict_check_decoders[decoder][-1]) != len(dict_ret[task]["decoders"][decoder]):
                            error_msg = "The ---'%s'--- output mis-match current 'Task %s-%s' vs. existing 'Task %s-%s'" \
                                        % (decoder, task, dict_ret[task]["decoders"][decoder],
                                           dict_check_decoders[decoder][0], dict_check_decoders[decoder][-1])
                            raise RuntimeError(error_msg)
                print_with_rank("Duplicated decoding head %s : Using 'Task %s-%s-%d' to UPDATE the existing 'Task %s-%s-%d'" %
                                (decoder, task, dict_ret[task]["decoders"][decoder], dict_ret[task]["train_order"], dict_check_decoders[decoder][0],
                                 dict_check_decoders[decoder][-1], dict_check_decoders[decoder][1]), rank=rank)
    # Making sure that the supporting decoder cannot 'reach to the future'.
    list_up_to_current_decoders = []
    for task in dict_ret:
        dict_ret[task] = sanity_check_supporting_organs(dict_ret[task], dict_check_decoders)
        dict_check_supporting = {}
        for decoder in dict_ret[task]["decoders"]:
            list_up_to_current_decoders.append(decoder)
        if dict_ret[task]["supporting"] is not None:
            for decoder in dict_ret[task]["supporting"]:
                if decoder not in list_up_to_current_decoders:
                    warnings_msg = "Decoding head that need 'Supporting' is not found: 'Task %s-%s'" % (task, decoder)
                    print(warnings_msg)
                else:
                    if len(dict_ret[task]["supporting"][decoder]) <= 0:
                        print("No supporting decoder is assigned to 'Task %s-%s'" % (task, decoder))
                    else:
                        for supporting in dict_ret[task]["supporting"][decoder]:
                            if supporting not in list_up_to_current_decoders:
                                print("Supporting head cannot be reached at the moment: Using '%s' to support 'Task %s-%s'" % (supporting, task, decoder))
                            else:
                                if decoder not in dict_check_supporting:
                                    dict_check_supporting[decoder] = [supporting]
                                else:
                                    dict_check_supporting[decoder].append(supporting)
        dict_ret[task]["supporting"] = dict_check_supporting
    # Get BPR ranges for decoders
    for task in dict_ret:
        bpr_file_preprocessing = join(preprocessing_output_dir, dict_ret[task]['task'], 'bpr_scores.json')
        # we need to make sure that the target pth exists.
        if not os.path.exists(dict_ret[task]["output_folder_name"]):
            try:
                os.makedirs(dict_ret[task]["output_folder_name"])
            except:
                print("Already exists: %s\n" % dict_ret[task]["output_folder_name"])
        bpr_file = join(dict_ret[task]["output_folder_name"], 'bpr_scores.json')
        if os.path.isfile(bpr_file_preprocessing):
            shutil.copy(bpr_file_preprocessing, bpr_file)
        flag_load_only_encoder = False
        if not os.path.exists(bpr_file):
            if "load_only_encoder" in dict_ret[task] and dict_ret[task]["load_only_encoder"]:
                flag_load_only_encoder = True
            else:
                raise RuntimeError("bpr_scores.json not found in either preprocessed folder or results folder!")
        bpr_per_task = None
        if not flag_load_only_encoder:
            bpr_per_task = json.load(open(bpr_file))
        dict_ret[task]['bpr_range_for_decoders'] = {}
        for decoder in dict_ret[task]['decoders']:
            dict_ret[task]['bpr_range_for_decoders'][decoder] = {}
            bottom_bpr = float("inf")
            top_bpr = float("-inf")
            max_bpr = float("-inf")
            min_bpr = float("inf")
            mean_bpr = float("-inf")
            std_bpr = float("-inf")
            task_cls = dict_ret[task]['decoders'][decoder]
            task_cls = [task_cls] if not isinstance(task_cls, list) else task_cls
            if not flag_load_only_encoder:
                for cls in task_cls:
                    if str(cls) in bpr_per_task:
                        bottom_bpr = np.nanmin([bpr_per_task[str(cls)]['percentile_00_5'], bottom_bpr])
                        top_bpr = np.nanmax([bpr_per_task[str(cls)]['percentile_99_5'], top_bpr])
                        min_bpr = np.nanmin([bpr_per_task[str(cls)]['mn'], min_bpr])
                        max_bpr = np.nanmax([bpr_per_task[str(cls)]['mx'], max_bpr])
                        mean_bpr = np.nanmax([bpr_per_task[str(cls)]['mean'], mean_bpr])
                        std_bpr = np.nanmax([bpr_per_task[str(cls)]['sd'], std_bpr])
                    else:
                        print("Warning, BPR range not found for Task -- '%s' class -- '%s'" % (task, cls))
            if bottom_bpr == float("inf"):
                bottom_bpr = float("-inf")
                top_bpr = float("inf")
                min_bpr = float("-inf")
                max_bpr = float("inf")
                mean_bpr = 0
                std_bpr = 0
            dict_ret[task]['bpr_range_for_decoders'][decoder]['percentile_00_5'] = bottom_bpr
            dict_ret[task]['bpr_range_for_decoders'][decoder]['percentile_99_5'] = top_bpr
            dict_ret[task]['bpr_range_for_decoders'][decoder]['min'] = min_bpr
            dict_ret[task]['bpr_range_for_decoders'][decoder]['max'] = max_bpr
            dict_ret[task]['bpr_range_for_decoders'][decoder]['mean'] = mean_bpr
            dict_ret[task]['bpr_range_for_decoders'][decoder]['std'] = std_bpr
    # finally, for each training task, the dict key order is first ordered using the following "key_order"
    # For the remaining keys, we order them alphabetically.
    key_order = ["task", "type", "train_order", "continue_training", "fold", "no_mirroring",
                 "pretrain_model_name", "load_only_encoder", "decoders", "supporting", "weights_for_decoders",
                 "bpr_range_for_decoders", "model_training_setup"]
    for task in dict_ret:
        current_keys = list(dict_ret[task].keys())
        for k in key_order:
            if k in current_keys:
                current_keys.remove(k)
        current_task = dict_ret[task]
        current_keys.sort()
        current_keys = key_order + current_keys
        reorder_dict = {}
        for k in current_keys:
            if k in current_task:
                reorder_dict[k] = current_task[k]
        dict_ret[task] = reorder_dict
    return trainer_class, dict_ret, network, trainer


def check_ge(task, dict_setup, dict_ret):
    default_general_encoder_setup = {
        "base_num_feature": default_base_num_feature,
        "num_conv_per_stage": None,
        "conv_kernel": None,
        "pool_kernel": None,
        "enable_ema": default_enable_encoder_ema,
    }
    dict_setup[task]["train_order"] = 0
    dict_ret[task] = dict_setup[task]
    dict_ret[task]["type"] = "GeneralEncoder"
    if "full_network" not in dict_ret[task]:
        dict_ret[task]["full_network"] = False
    # If the decoders are not set in cfg.json, we set "load_only_encoder" to True.
    # Else, we try to load the decoders and let "load_only_encoder" be False
    if "decoders" not in dict_ret[task] or len(dict_ret[task]["decoders"]) == 0:
        dict_ret[task]["load_only_encoder"] = True
    else:
        dict_ret[task]["load_only_encoder"] = False

    dict_ret[task] = dict_attribute_check(dict_ret[task], task)
    dict_ret[task] = get_model_training_setup(dict_ret[task])

    if "encoder_architecture_setup" not in dict_ret[task]:
        dict_ret[task]["encoder_architecture_setup"] = default_general_encoder_setup
    else:
        if "num_conv_per_stage" not in dict_ret[task]["encoder_architecture_setup"]:
            dict_ret[task]["encoder_architecture_setup"]["num_conv_per_stage"] = None
        if "conv_kernel" not in dict_ret[task]["encoder_architecture_setup"] \
                or len(dict_ret[task]["encoder_architecture_setup"]["conv_kernel"]) != 6:
            dict_ret[task]["encoder_architecture_setup"]["conv_kernel"] = None
        if "pool_kernel" not in dict_ret[task]["encoder_architecture_setup"] \
                or len(dict_ret[task]["encoder_architecture_setup"]["pool_kernel"]) != 5:
            dict_ret[task]["encoder_architecture_setup"]["pool_kernel"] = default_pool
        if "base_num_feature" not in dict_ret[task]["encoder_architecture_setup"]:
            dict_ret[task]["encoder_architecture_setup"]["base_num_feature"] = default_base_num_feature
        if "conv_block" not in dict_ret[task]["encoder_architecture_setup"]:
            dict_ret[task]["encoder_architecture_setup"]["conv_block"] = default_ge_basic_block
        if "enable_ema" not in dict_ret[task]["encoder_architecture_setup"]:
            dict_ret[task]["encoder_architecture_setup"]["enable_ema"] = default_enable_encoder_ema

    # Store the General Encoder settings in "bak",
    # s.t., during network initialization, the decoders/supporting heads are not included.
    if dict_ret[task]["load_only_encoder"]:
        dict_ret[task]["bak"] = {}
        dict_ret[task]["bak"]["decoders"] = dict_ret[task]["decoders"]
        dict_ret[task]["bak"]["supporting"] = dict_ret[task]["supporting"]
        dict_ret[task]["bak"]["weights_for_decoders"] = dict_ret[task]["weights_for_decoders"]
        dict_ret[task]["bak"]["model_training_setup"] = dict_ret[task]["model_training_setup"]
        # Set the "decoders" and "supporting" to EMPTY, so that the network will not load them.
        dict_ret[task]["decoders"] = {}
        dict_ret[task]["supporting"] = {}
        dict_ret[task]["weights_for_decoders"] = {}
        dict_ret[task]["model_training_setup"]["decoders"] = None
        dict_ret[task]["model_training_setup"]["supporting"] = None


def sanity_check_supporting_organs(train_dict, decoder_dict):
    """
    reorder the supporing organ training sequence
    Making sure that the most used supporting organs are prioritized
    E.g.,
    Before reorder:
    {"sh": ["anchor", "mid"], "mid": ["anchor"]}
    After reorder:
    {"mid": ["anchor"], "sh": ["anchor", "mid"]}
    """

    # Define functions to find the supporting loops using DFS.
    def dfs(visited_organs, dict, organ):
        if organ in visited_organs:
            return organ
        elif organ not in dict:
            return None
        else:
            visited_organs.add(organ)
            for supp_organ in dict[organ]:
                if dfs(visited_organs, dict, supp_organ) is not None:
                    return supp_organ
            return None

    supporting_dict = train_dict["supporting"]
    if supporting_dict is not None:
        # 1st: Remove non-exist supporting and self-supporting organs, e.g., "sh":["anchor", "sh"]
        for organ in supporting_dict:
            cleaned_supporting_organs = []
            for supporting_organ in supporting_dict[organ]:
                if organ != supporting_organ:
                    # if supporting_organ in train_dict["decoders"]:
                    if supporting_organ in decoder_dict:
                        cleaned_supporting_organs.append(supporting_organ)
                    else:
                        print("Warning non-exist supporting: Using '%s' to support '%s'" % (supporting_organ, organ))
                else:
                    print("Warning self-supporting: Using '%s' to support '%s'" % (supporting_organ, organ))
            supporting_dict[organ] = cleaned_supporting_organs

        # 2nd: Remove the "organ_needs_supporting", which contains loops.
        organs_need_supporting_list = list(supporting_dict.keys())
        removed_supporting_list = []
        for organ in organs_need_supporting_list:
            if organ not in removed_supporting_list:
                visited_organs = set()
                loop_organ = dfs(visited_organs, supporting_dict, organ)
                if loop_organ is not None:
                    print("Loop found in SUPPORTING-PATH: Using '%s' to support '%s'" % (loop_organ, organ))
                    del supporting_dict[organ]
                    removed_supporting_list.append(organ)

        # 3rd: Re-order the supporting dictionary considering the referencing times.
        organ_cross_referenced_times = {}
        for organ in supporting_dict:
            organ_cross_referenced_times[organ] = 0
        organs_need_supporting_list = list(supporting_dict.keys())
        for organ in organs_need_supporting_list:
            for rest_organ in supporting_dict:
                if organ != rest_organ:
                    for supporting_organ in supporting_dict[rest_organ]:
                        if organ == supporting_organ:
                            organ_cross_referenced_times[organ] += 1
        cross_referenced_times = OrderedDict(sorted(organ_cross_referenced_times.items(),
                                                    key=lambda x: x[1], reverse=True))
        supporting_dict = {k: supporting_dict[k] for k in cross_referenced_times.keys()}

        train_dict["supporting"] = supporting_dict
    return train_dict


def parse_training_setup(current_training_setup):
    """
    Try to parse the training setup for each decoding head.
    """
    if current_training_setup is not None:
        # If the input is a list.
        # Then, we try to match the pre-defined order of [decay_lower, decay_upper, lr, epoch, load_pretrain, prune]
        if isinstance(current_training_setup, list):
            if len(current_training_setup) < 2:
                current_foreground_sampling_decay = default_sampling_decay
                current_foreground_sampling_decay = list(sorted(current_foreground_sampling_decay, reverse=True))
                current_lr_epoch_load_ema = [None, None, False, False, False]
            else:
                current_foreground_sampling_decay = current_training_setup[:2]
                current_foreground_sampling_decay = list(sorted(current_foreground_sampling_decay, reverse=True))
                current_foreground_sampling_decay = np.clip(current_foreground_sampling_decay, 0, 1)
                current_lr_epoch_load_ema = current_training_setup[2:7]
                current_lr_epoch_load_ema = try_to_correct_lr_ep_load_prune_ema(current_lr_epoch_load_ema)
            return current_foreground_sampling_decay, current_lr_epoch_load_ema
        # If the input is a dict.
        # Then, we try to parse each input using keywords: foreground_sampling_decay, lr, epoch, load/load_pretrain, prune
        elif isinstance(current_training_setup, dict):
            if "foreground_sampling_decay" in current_training_setup:
                current_foreground_sampling_decay = current_training_setup["foreground_sampling_decay"]
                current_foreground_sampling_decay.sort(reverse=True)
            else:
                current_foreground_sampling_decay = default_sampling_decay
                current_foreground_sampling_decay = list(sorted(current_foreground_sampling_decay, reverse=True))
            foreground_sampling_decay = np.clip(current_foreground_sampling_decay, 0, 1)

            current_lr_epoch_load = [None, default_max_epoch, default_load_from_decoder, default_prune_decoder, default_enable_decoder_ema]
            if "lr" in current_training_setup:
                if isinstance(current_training_setup["lr"], (int, float)):
                    current_lr_epoch_load[0] = float(current_training_setup["lr"])

            if "epoch" in current_training_setup:
                if isinstance(current_training_setup["epoch"], (int, float)):
                    # current_lr_epoch_load[1] = max(default_pruning_percentile_moving_average_window_size + 1, int(current_training_setup["epoch"]))
                    current_lr_epoch_load[1] = max(1, int(current_training_setup["epoch"]))
                    if current_lr_epoch_load[0] is None:
                        current_lr_epoch_load[0] = default_lr

            if "load_pretrain" in current_training_setup:
                current_lr_epoch_load[2] = bool(current_training_setup["load_pretrain"])
            if "load" in current_training_setup:
                current_lr_epoch_load[2] = bool(current_training_setup["load"])

            if "prune" in current_training_setup:
                current_lr_epoch_load[3] = bool(current_training_setup["prune"])
                # If only "prune" is set, then we set the lr = default_lr.
                if "lr" not in current_training_setup:
                    current_lr_epoch_load[0] = default_lr
                if "epoch" not in current_training_setup:
                    current_lr_epoch_load[1] = default_max_epoch

            if "ema" in current_training_setup:
                current_lr_epoch_load[4] = bool(current_training_setup["ema"])
            if "enable_ema" in current_training_setup:
                current_lr_epoch_load[4] = bool(current_training_setup["enable_ema"])

            current_lr_epoch_load = try_to_correct_lr_ep_load_prune_ema(current_lr_epoch_load)
            return foreground_sampling_decay, current_lr_epoch_load
        else:
            return list(sorted(default_sampling_decay, reverse=True)), [None, None, False, False, False]


def try_to_correct_lr_ep_load_prune_ema(decoder):
    """
    We try to check 1) learning rate, 2) total training epochs, and 3) if-or-not loading pretrained weights.
    """
    # If the decoder is None, then, we set all to None, but "try to" load pretrained weights
    # The "trainer" will detection if the pretrained weights are available.
    # If the pretrained weights are not available, then the "trainer" will re-train it.
    if decoder is None or len(decoder) == 0:
        decoder = [None, None, True]
        return decoder

    # If the LR is less than 0, then we set LR=0.01, EPOCHS=1000, LOAD=False
    # If LR is between [0, 2), then we set EPOCHS=pre-defined EPOCHS, LOAD=pre-defined load
    # If LR larger than [2, inf), then we set LR=0.01, EPOCHS=1000, LOAD=False
    # If LR is None or not-a-number, then we set LR=EPOCHS=None and LOAD=False
    def check_lr(current_decoder):
        lr = current_decoder[0]
        if isinstance(lr, (int, float)):
            if lr < 0 or lr >= 2:
                current_decoder = [default_lr, default_max_epoch, default_load_from_decoder, default_prune_decoder, default_enable_decoder_ema]
            else:
                if len(current_decoder) == 5:
                    current_decoder = \
                        [float(lr), max(1, int(current_decoder[1])), bool(current_decoder[2]), bool(current_decoder[3]), bool(current_decoder[4])]
                elif len(current_decoder) == 4:
                    current_decoder = \
                        [float(lr), max(1, int(current_decoder[1])), bool(current_decoder[2]), bool(current_decoder[3]), default_enable_decoder_ema]
                elif len(current_decoder) == 3:
                    current_decoder = \
                        [float(lr), max(1, int(current_decoder[1])), bool(current_decoder[2]), default_prune_decoder, default_enable_decoder_ema]
                elif len(current_decoder) == 2:
                    current_decoder = \
                        [float(lr), max(1, int(current_decoder[1])), default_load_from_decoder, default_prune_decoder, default_enable_decoder_ema]
                else:
                    current_decoder = \
                        [float(lr), max(1, default_max_epoch), default_load_from_decoder, default_prune_decoder, default_enable_decoder_ema]
                # current_decoder[1] = max(current_decoder[1], default_pruning_percentile_moving_average_window_size + 1)
                current_decoder[1] = current_decoder[1]
        else:
            current_decoder = [None, None, False, False, False]
        return current_decoder

    # The input "current_decoder" must have 2 elements. We assume the LR is already corrected.
    # If EPOCHS is less than 0, then we set LR=EPOCHS=None, LOAD=True
    # If EPOCHS is between [0, 2), then we assume the input is wrong and set EPOCHS=1000
    # If EPOCHS is None or not-a-number, then we set LR=EPOCHS=None, LOAD=False
    def check_epoch(current_decoder):
        ep = current_decoder[1]
        if isinstance(ep, (int, float)):
            if ep < 0:
                current_decoder = [None, None, False, False, False]
            else:
                if ep < 1:
                    current_decoder[1] = default_max_epoch
        else:
            current_decoder = [None, None, False, False, False]
        return current_decoder

    # The input "current_decoder" must have 3 elements. We assume the LR, EPOCHS are already corrected.
    # If LOAD is not a boolean, then we set LOAD=False
    def check_load(current_decoder):
        lo = current_decoder[2]
        if not isinstance(lo, bool):
            current_decoder[2] = False
        return current_decoder

    # If the input has only 1 number, we ASSUME the input number is LR
    if len(decoder) == 1:
        return check_lr(decoder)

    # If the input has two numbers, we ASSUME the input numbers are LR and EPOCHS
    if len(decoder) == 2:
        decoder = check_lr(decoder)
        return check_epoch(decoder)

    if len(decoder) > 2:
        decoder = check_lr(decoder)
        decoder = check_epoch(decoder)
        return check_load(decoder)


def get_model_training_setup(dict_in):
    if "decoders_to_train" not in dict_in:
        dict_in["decoders_to_train"] = []
    if "model_training_setup" not in dict_in:
        warnings.warn("Attribute 'model_training_setup' is not found!")
        dict_in["model_training_setup"] = {}
    if "decoders" not in dict_in["model_training_setup"]:
        warnings.warn("Attribute 'model_training_setup.decoders' is not found!")
        dict_in["model_training_setup"]["decoders"] = None
    if "supporting" not in dict_in["model_training_setup"]:
        warnings.warn("Attribute 'model_training_setup.supporting' is not found!")
        dict_in["model_training_setup"]["supporting"] = None
    if "patch_size" not in dict_in["model_training_setup"]:
        dict_in["model_training_setup"]["patch_size"] = {}
    if "batch_size" not in dict_in["model_training_setup"]:
        dict_in["model_training_setup"]["batch_size"] = {}
    model_training_setup = dict_in["model_training_setup"]
    decoders_training_setup = model_training_setup["decoders"]
    decoders_to_train = dict_in["decoders_to_train"]
    support_training_setup = model_training_setup["supporting"]
    # patch_size = model_training_setup["patch_size"]
    # initialization
    dict_parse = {"decoders": {}, "supporting": {}, "patch_size": {}, "batch_size": {}}
    # parse decoder's LR and training EPOCHS
    if decoders_training_setup is not None and isinstance(decoders_training_setup, dict):
        for key in decoders_training_setup:
            foreground_sampling_decay, lr_epoch_load_prune_ema = parse_training_setup(decoders_training_setup[key])
            if key == "all":
                dict_parse["decoders"] = {}
                dict_parse["decoders"]["all"] = list(foreground_sampling_decay) + list(lr_epoch_load_prune_ema)
                for decoder in dict_in["decoder_architecture_setup"]:
                    if lr_epoch_load_prune_ema[-1]:
                        if not dict_in["decoder_architecture_setup"][decoder]["enable_ema"]:
                            dict_in["decoder_architecture_setup"][decoder]["enable_ema"] = True
                break
            else:
                if key in dict_in["decoders"]:
                    dict_parse["decoders"][key] = list(foreground_sampling_decay) + list(lr_epoch_load_prune_ema)
                    if lr_epoch_load_prune_ema[-1] and key in dict_in["decoder_architecture_setup"]:
                        if not dict_in["decoder_architecture_setup"][key]["enable_ema"]:
                            dict_in["decoder_architecture_setup"][key]["enable_ema"] = True
    else:
        dict_parse["decoders"] = None

    # parse support's LR and training EPOCHS
    if support_training_setup is not None and isinstance(support_training_setup, dict):
        for key in support_training_setup:
            foreground_sampling_decay, lr_epoch_load_prune_ema = parse_training_setup(support_training_setup[key])
            if key == "all":
                dict_parse["supporting"] = {}
                dict_parse["supporting"]["all"] = list(foreground_sampling_decay) + list(lr_epoch_load_prune_ema)
                for decoder in dict_in["decoder_architecture_setup"]:
                    if lr_epoch_load_prune_ema[-1]:
                        if not dict_in["decoder_architecture_setup"][decoder]["enable_ema"]:
                            dict_in["decoder_architecture_setup"][decoder]["enable_ema"] = True
                break
            else:
                if key in dict_in["supporting"]:
                    dict_parse["supporting"][key] = list(foreground_sampling_decay) + list(lr_epoch_load_prune_ema)
                    if lr_epoch_load_prune_ema[-1] and key in dict_in["decoder_architecture_setup"]:
                        if not dict_in["decoder_architecture_setup"][key]["enable_ema"]:
                            dict_in["decoder_architecture_setup"][key]["enable_ema"] = True
    else:
        if dict_in["supporting"] is not None and len(dict_in["supporting"]) > 0:
            if dict_parse["supporting"] is None:
                dict_parse["supporting"] = {}
            for supporting in dict_in["supporting"]:
                if dict_parse["decoders"] is not None and supporting in dict_parse["decoders"]:
                    dict_parse["supporting"][supporting] = copy.deepcopy(dict_parse["decoders"][supporting])
        else:
            dict_parse["supporting"] = None
    # We try to check if pre-defined "decoders_to_train" is properly defined. If not, we set it to default.
    default_training_setup = list(sorted(default_sampling_decay, reverse=True)) + [default_lr, default_max_epoch, default_load_from_decoder,
                                                                                   default_prune_decoder, default_enable_decoder_ema]
    for decoder in decoders_to_train:
        if decoder.lower() == "all":
            if dict_in["decoders"] is not None:
                for decoder_in_dict in dict_in["decoders"]:
                    if dict_parse["decoders"] is None:
                        dict_parse["decoders"] = {}
                    if decoder_in_dict not in dict_parse["decoders"]:
                        dict_parse["decoders"][decoder_in_dict] = copy.deepcopy(default_training_setup)
            if dict_in["supporting"] is not None:
                for supporting_in_dict in dict_in["supporting"]:
                    if dict_parse["supporting"] is None:
                        dict_parse["supporting"] = {}
                    if dict_parse["decoders"] is not None and supporting_in_dict in dict_parse["decoders"] and \
                            dict_parse["decoders"][supporting_in_dict] is not None and len(dict_parse["decoders"][supporting_in_dict]) > 0:
                        dict_parse["supporting"][supporting_in_dict] = copy.deepcopy(dict_parse["decoders"][supporting_in_dict])
                    if supporting_in_dict not in dict_parse["supporting"] and "all" not in dict_parse["supporting"]:
                        dict_parse["supporting"][supporting_in_dict] = copy.deepcopy(default_training_setup)
        else:
            if dict_in["decoders"] is not None and len(dict_in["decoders"]) > 0 and decoder in dict_in["decoders"]:
                if dict_parse["decoders"] is None:
                    dict_parse["decoders"] = {}
                if decoder not in dict_parse["decoders"] and "all" not in dict_parse["decoders"]:
                    dict_parse["decoders"][decoder] = copy.deepcopy(default_training_setup)
            if dict_in["supporting"] is not None and len(dict_in["supporting"]) > 0 and decoder in dict_in["supporting"]:
                if dict_parse["supporting"] is None:
                    dict_parse["supporting"] = {}
                if dict_parse["decoders"] is not None and decoder in dict_parse["decoders"] and \
                        dict_parse["decoders"][decoder] is not None and len(dict_parse["decoders"][decoder]) > 0:
                    dict_parse["supporting"][decoder] = copy.deepcopy(dict_parse["decoders"][decoder])
                if decoder not in dict_parse["supporting"] and "all" not in dict_parse["supporting"]:
                    dict_parse["supporting"][decoder] = copy.deepcopy(default_training_setup)
    # We try to remove the repeated training setups: If supporting is already set, then we removed the training setup
    if dict_parse["supporting"] is not None and len(dict_parse["supporting"]) > 0:
        for decoder in dict_parse["supporting"]:
            if decoder in dict_parse["decoders"]:
                del dict_parse["decoders"][decoder]
        if "all" in dict_parse["supporting"]:
            for decoder in dict_in["supporting"]:
                if decoder in dict_parse["decoders"]:
                    del dict_parse["decoders"][decoder]
    # We try to check the pre-defined patch size for each decoding head
    for decoder in dict_in["decoders"]:
        if decoder not in dict_in["model_training_setup"]["patch_size"]:
            dict_parse["patch_size"][decoder] = None
        else:
            # The minimum dimension is 2^0=1 (e.g., patch size = 8x16x16), such that the bottom feature dim is 1x1x1
            # might cause skip-feature vs. decoding-feature mismatch -> We use "interpolation" to "match" the dimension.
            pool_scales = np.prod(default_pool, 0) / 2
            dict_parse["patch_size"][decoder] = copy.deepcopy(default_patch_size)
            if len(dict_in["model_training_setup"]["patch_size"][decoder]) == 3:
                for i in range(3):
                    # The input size must be a shape of 2^* times of an integer.
                    if abs(dict_in["model_training_setup"]["patch_size"][decoder][i]) % pool_scales[i] != 0:
                        tmp_ratio = max(1, int(abs(dict_in["model_training_setup"]["patch_size"][decoder][i]) / pool_scales[i]))
                        dict_parse["patch_size"][decoder][i] = int(tmp_ratio * pool_scales[i])
                    else:
                        if dict_in["model_training_setup"]["patch_size"][decoder][i] == 0:
                            dict_parse["patch_size"][decoder][i] = abs(int(default_patch_size[i]))
                        else:
                            dict_parse["patch_size"][decoder][i] = abs(int(dict_in["model_training_setup"]["patch_size"][decoder][i]))

    # if the "all" is not defined in cfg, then we set the largest patch size considering all decoding paths.
    if "all" not in dict_in["model_training_setup"]["patch_size"]:
        dict_parse["patch_size"]["all"] = None
    else:
        if dict_in["model_training_setup"]["patch_size"]["all"] is None:
            patch_size_all = default_patch_size
            for decoder in dict_in["model_training_setup"]["patch_size"]:
                if dict_in["model_training_setup"]["patch_size"][decoder] is not None:
                    for d in range(len(patch_size_all)):
                        patch_size_all[d] = max(patch_size_all[d], dict_in["model_training_setup"]["patch_size"][decoder][d])
        dict_parse["patch_size"]["all"] = dict_in["model_training_setup"]["patch_size"]["all"]

    # We try to check the pre-defined batch size for each decoding head
    for decoder in dict_in["decoders"]:
        if decoder not in dict_in["model_training_setup"]["batch_size"]:
            dict_parse["batch_size"][decoder] = copy.deepcopy(default_batch_size)
        else:
            if not isinstance(dict_in["model_training_setup"]["batch_size"][decoder], int):
                dict_parse["batch_size"][decoder] = copy.deepcopy(default_batch_size)
            else:
                dict_parse["batch_size"][decoder] = dict_in["model_training_setup"]["batch_size"][decoder]
    # if the "all" is not defined in cfg, then we set the largest batch size to be the default batch size
    if "all" not in dict_in["model_training_setup"]["batch_size"]:
        dict_parse["batch_size"]["all"] = default_batch_size
    else:
        dict_parse["batch_size"]["all"] = dict_in["model_training_setup"]["batch_size"]["all"]
    dict_in["model_training_setup"] = dict_parse
    return dict_in


def dict_attribute_check(dict_to_check, task):
    keys_must_have = ["task", "decoders"]
    for key in keys_must_have:
        if key == "decoders":
            # Check if the "load_only_encoder" is defined in cfg.json
            if "load_only_encoder" in dict_to_check and dict_to_check["load_only_encoder"]:
                if "decoders" not in dict_to_check:
                    dict_to_check["decoders"] = {}
            for decoder in dict_to_check["decoders"]:
                if isinstance(dict_to_check["decoders"][decoder], str):
                    tmp_idx = []
                    ranges = dict_to_check["decoders"][decoder].split(",")
                    for r in ranges:
                        r = r.strip()
                        if "-" in r:
                            start, end = map(int, r.split("-"))
                            tmp_idx.extend(range(start, end + 1))
                        elif "_" in r:
                            start, end = map(int, r.split("_"))
                            tmp_idx.extend(range(start, end + 1))
                        else:
                            tmp_idx.append(int(r))
                    dict_to_check["decoders"][decoder] = list(set(tmp_idx))
                    if len(dict_to_check["decoders"][decoder]) == 0:
                        error_msg = "Decoding head Task %s - %s is not defined" % (task, decoder)
                        raise RuntimeError(error_msg)
                    if len(dict_to_check["decoders"][decoder]) == 1:
                        dict_to_check["decoders"][decoder] = dict_to_check["decoders"][decoder][0]
                elif isinstance(dict_to_check["decoders"][decoder], list):
                    # Convert all idxes to integers
                    tmp_idx = []
                    for idx in dict_to_check["decoders"][decoder]:
                        tmp_idx.append(int(idx))
                    # remove duplicated indices
                    dict_to_check["decoders"][decoder] = list(set(tmp_idx))
                    if len(dict_to_check["decoders"][decoder]) == 0:
                        error_msg = "Decoding head Task %s - %s is not defined" % (task, decoder)
                        raise RuntimeError(error_msg)
                    if len(dict_to_check["decoders"][decoder]) == 1:
                        dict_to_check["decoders"][decoder] = dict_to_check["decoders"][decoder][0]
                else:
                    dict_to_check["decoders"][decoder] = int(dict_to_check["decoders"][decoder])
        else:
            if key not in dict_to_check:
                raise RuntimeError("Key %s is not found in %s" % (key, dict_to_check))
    # keys_could_be_ignored:
    # "fold", "supporting", "model_training_setup", "no_mirroring", "continue_training", "pretrain_model_name",
    # "optimizer", "save_npz", "disable_saving", "val_disable_overwrite",  "run_mixed_precision", "decompress_data",
    # "disable_postprocessing_on_folds", "deterministic", "oversample_fg_percent", "disable_validation"
    if "finetune_encoder" not in dict_to_check or not isinstance(dict_to_check["finetune_encoder"], bool):
        dict_to_check["finetune_encoder"] = False
    if "plot_network" not in dict_to_check or not isinstance(dict_to_check["plot_network"], bool):
        dict_to_check["plot_network"] = True

    if "save_npz" not in dict_to_check or not isinstance(dict_to_check["save_npz"], bool):
        dict_to_check["save_npz"] = False

    if "disable_saving" not in dict_to_check or not isinstance(dict_to_check["disable_saving"], bool):
        dict_to_check["disable_saving"] = False

    if "val_disable_overwrite" not in dict_to_check or not isinstance(dict_to_check["val_disable_overwrite"], bool):
        dict_to_check["val_disable_overwrite"] = True

    if "run_mixed_precision" not in dict_to_check or not isinstance(dict_to_check["run_mixed_precision"], bool):
        dict_to_check["run_mixed_precision"] = True

    if "decompress_data" not in dict_to_check or not isinstance(dict_to_check["decompress_data"], bool):
        dict_to_check["decompress_data"] = True

    if "deterministic" not in dict_to_check or not isinstance(dict_to_check["deterministic"], bool):
        dict_to_check["deterministic"] = False

    if "disable_postprocessing_on_folds" not in dict_to_check or not isinstance(dict_to_check["disable_postprocessing_on_folds"], bool):
        dict_to_check["disable_postprocessing_on_folds"] = False

    if "disable_validation" not in dict_to_check or not isinstance(dict_to_check["disable_validation"], bool):
        dict_to_check["disable_validation"] = True

    if "warmup" not in dict_to_check:
        dict_to_check["warmup"] = default_warmup_epoch
    elif dict_to_check["warmup"] is None:
        dict_to_check["warmup"] = 0
    elif isinstance(dict_to_check["warmup"], (int, float)):
        dict_to_check["warmup"] = float(dict_to_check["warmup"])
    else:
        dict_to_check["warmup"] = 0

    if "amsgrad" not in dict_to_check:
        dict_to_check["amsgrad"] = True

    if "optimizer" not in dict_to_check or not isinstance(dict_to_check["fold"], (str, int)):
        dict_to_check["optimizer"] = default_optimizer
    else:
        dict_to_check["optimizer"] = dict_to_check["optimizer"].lower()
        if "sgd" in dict_to_check["optimizer"]:
            dict_to_check["optimizer"] = "sgd"
        elif "adam" in dict_to_check["optimizer"]:
            if "adamw" in dict_to_check["optimizer"]:
                dict_to_check["optimizer"] = "adamw"
            else:
                dict_to_check["optimizer"] = "adam"

    if "fold" not in dict_to_check or not isinstance(dict_to_check["fold"], (str, int)):
        dict_to_check["fold"] = 0
    else:
        if isinstance(dict_to_check["fold"], str) and "all" in dict_to_check["fold"].lower():
            dict_to_check["fold"] = "all"
        else:
            dict_to_check["fold"] = int(dict_to_check["fold"])

    if "supporting" not in dict_to_check:
        dict_to_check["supporting"] = None

    # We check three items here: learning rate, training epochs, if-or-not loading the pretrained weights.
    if "model_training_setup" not in dict_to_check:
        dict_to_check["model_training_setup"] = {}
        dict_to_check["model_training_setup"]["decoders"] = None
        dict_to_check["model_training_setup"]["supporting"] = None
        dict_to_check["model_training_setup"]["patch_size"] = {}
    else:
        if "decoders" not in dict_to_check["model_training_setup"]:
            dict_to_check["model_training_setup"]["decoders"] = None

        if "supporting" not in dict_to_check["model_training_setup"]:
            dict_to_check["model_training_setup"]["supporting"] = None

        if "patch_size" not in dict_to_check["model_training_setup"]:
            dict_to_check["model_training_setup"]["patch_size"] = {}

    # We check if the decoder_architecture_setup is pre-defined in the "cfg_training.json" file
    if "decoder_architecture_setup" in dict_to_check and isinstance(dict_to_check["decoder_architecture_setup"], dict):
        for decoder in dict_to_check["decoders"]:
            # Dict in python is in-place
            if "load_only_encoder" in dict_to_check and dict_to_check["load_only_encoder"]:
                dict_to_check["decoder_architecture_setup"][decoder] = None
            else:
                tmp_decoder_architecture = {"base_num_feature": get_recommend_base_num_features(dict_to_check["decoders"][decoder]),
                                            "num_conv_per_stage": get_recommend_num_conv_per_stage(dict_to_check["decoders"][decoder]),
                                            "conv_kernel": None, "enable_ema": default_enable_decoder_ema,
                                            "conv_block": default_decoder_basic_block}
                if decoder in dict_to_check["decoder_architecture_setup"]:
                    if isinstance(dict_to_check["decoder_architecture_setup"][decoder], list):
                        # check if the architecture setup is in "list" short setup version
                        if len(dict_to_check["decoder_architecture_setup"][decoder]) > 1:
                            # the first element is the "base_num_feature"
                            tmp_decoder_architecture["base_num_feature"] = int(dict_to_check["decoder_architecture_setup"][decoder][0])
                        if len(dict_to_check["decoder_architecture_setup"][decoder]) > 2:
                            # The second element is the "num_conv_per_stage"
                            tmp_decoder_architecture["num_conv_per_stage"] = int(dict_to_check["decoder_architecture_setup"][decoder][1])
                        if 3 < len(dict_to_check["decoder_architecture_setup"][decoder]) <= 7:
                            # The third to the seventh element are the "conv_kernel"
                            for i, tmp_kernel in enumerate(dict_to_check["decoder_architecture_setup"][decoder][2:7]):
                                if isinstance(tmp_kernel, list) and len(tmp_kernel) == 3:
                                    tmp_decoder_architecture["conv_kernel"][i] = tmp_kernel
                        if len(dict_to_check["decoder_architecture_setup"][decoder]) > 7:
                            # The last/eighth element is the boolean value for "enable_ema"
                            tmp_decoder_architecture["enable_ema"] = bool(dict_to_check["decoder_architecture_setup"][decoder][7])
                        if len(dict_to_check["decoder_architecture_setup"][decoder]) > 8:
                            tmp_decoder_architecture["conv_block"] = default_decoder_basic_block
                    elif isinstance(dict_to_check["decoder_architecture_setup"][decoder], dict):
                        # check if the architecture setup is in "dict" full setup version
                        if "base_num_feature" in dict_to_check["decoder_architecture_setup"][decoder]:
                            tmp_decoder_architecture["base_num_feature"] = int(dict_to_check["decoder_architecture_setup"][decoder]["base_num_feature"])
                        if "num_conv_per_stage" in dict_to_check["decoder_architecture_setup"][decoder]:
                            tmp_decoder_architecture["num_conv_per_stage"] = int(dict_to_check["decoder_architecture_setup"][decoder]["num_conv_per_stage"])
                        if "conv_kernel" in dict_to_check["decoder_architecture_setup"][decoder]:
                            if dict_to_check["decoder_architecture_setup"][decoder] is not None and \
                                    len(dict_to_check["decoder_architecture_setup"][decoder]) >= 5:
                                for i, tmp_kernel in enumerate(dict_to_check["decoder_architecture_setup"][decoder]["conv_kernel"][:5]):
                                    if isinstance(tmp_kernel, list) and len(tmp_kernel) == 3:
                                        tmp_decoder_architecture["conv_kernel"][i] = tmp_kernel
                        if "enable_ema" in dict_to_check["decoder_architecture_setup"][decoder]:
                            tmp_decoder_architecture["enable_ema"] = bool(dict_to_check["decoder_architecture_setup"][decoder]["enable_ema"])
                        if "conv_block" in dict_to_check["decoder_architecture_setup"][decoder]:
                            tmp_decoder_architecture["conv_block"] = dict_to_check["decoder_architecture_setup"][decoder]["conv_block"]
                dict_to_check["decoder_architecture_setup"][decoder] = tmp_decoder_architecture
                del tmp_decoder_architecture
    else:
        # If the 1) "decoder" is not EMPTY, 2) "decoder_architecture_setup" is NOT in the cfg.json, and 3) "load_only_encoder" is not True
        # Then, we initialize all decoders using the default setup
        dict_to_check["decoder_architecture_setup"] = {}
        for decoder in dict_to_check["decoders"]:
            if "load_only_encoder" in dict_to_check and dict_to_check["load_only_encoder"]:
                dict_to_check["decoder_architecture_setup"][decoder] = None
            else:
                dict_to_check["decoder_architecture_setup"][decoder] = \
                    {"base_num_feature": get_recommend_base_num_features(dict_to_check["decoders"][decoder]),
                     "num_conv_per_stage": get_recommend_num_conv_per_stage(dict_to_check["decoders"][decoder]),
                     "conv_kernel": None, "enable_ema": default_enable_decoder_ema, "conv_block": default_decoder_basic_block}
    # Try to parse if the training involve "mirroring" data aug
    if "mirroring" not in dict_to_check or not isinstance(dict_to_check["mirroring"], bool):
        dict_to_check["no_mirroring"] = True
    if "no_mirroring" not in dict_to_check or not isinstance(dict_to_check["no_mirroring"], bool):
        dict_to_check["no_mirroring"] = True
    if "mirroring" in dict_to_check and "no_mirroring" in dict_to_check:
        dict_to_check["no_mirroring"] = bool(dict_to_check["no_mirroring"])

    if "continue_training" not in dict_to_check or not isinstance(dict_to_check["continue_training"], bool):
        dict_to_check["continue_training"] = False

    if "pretrain_model_name" not in dict_to_check:
        dict_to_check["pretrain_model_name"] = None
    else:
        # Try to locate the model in pre-defined pth. If it does not exist, then try to use keywords for referencing
        if dict_to_check["pretrain_model_name"] is not None:
            if not os.path.exists(dict_to_check["pretrain_model_name"]):
                if "final" in dict_to_check["pretrain_model_name"].lower():
                    dict_to_check["pretrain_model_name"] = "model_final_checkpoint.model"
                elif "best" in dict_to_check["pretrain_model_name"].lower():
                    dict_to_check["pretrain_model_name"] = "model_best.model"
                elif "latest" in dict_to_check["pretrain_model_name"].lower():
                    dict_to_check["pretrain_model_name"] = "model_latest.model"
                else:
                    dict_to_check["pretrain_model_name"] = None

    # Get the weights for the 'training' decoding heads
    # We only balance the weight when all heads are in training.
    # We do not need to balance the weight if we update each head individually
    if "weights_for_decoders" not in dict_to_check or not isinstance(dict_to_check["weights_for_decoders"], dict):
        dict_to_check["weights_for_decoders"] = {}
    # Initialize the weights using the predefined weight, otherwise using the number of output classes.
    for decoder in dict_to_check["decoders"]:
        if decoder not in dict_to_check["weights_for_decoders"]:
            if isinstance(dict_to_check["decoders"][decoder], list):
                dict_to_check["weights_for_decoders"][decoder] = len(dict_to_check["decoders"][decoder])
            else:
                dict_to_check["weights_for_decoders"][decoder] = 1
    decoder_weight = {}
    # Remove the non-existing decoder's weight
    for decoder in dict_to_check["weights_for_decoders"]:
        if decoder in dict_to_check["decoders"]:
            decoder_weight[decoder] = dict_to_check["weights_for_decoders"][decoder]
    dict_to_check["weights_for_decoders"] = decoder_weight
    weight_for_decoder = None
    if dict_to_check["model_training_setup"]["decoders"] is not None:
        for decoder in dict_to_check["model_training_setup"]["decoders"]:
            if decoder == "all":
                weight_for_decoder = dict_to_check["weights_for_decoders"].copy()
        if weight_for_decoder is not None:
            weight_sum_decoder = 1.0 * sum(weight_for_decoder.values())
            for decoder in weight_for_decoder:
                weight_for_decoder[decoder] /= weight_sum_decoder
    weight_for_support = None
    if dict_to_check["model_training_setup"]["supporting"] is not None:
        for decoder in dict_to_check["model_training_setup"]["supporting"]:
            if decoder == "all":
                weight_for_support = dict_to_check["weights_for_decoders"].copy()
        if weight_for_support is not None:
            weight_sum_support = 1.0 * sum(weight_for_support.values())
            for decoder in weight_for_support:
                weight_for_support[decoder] /= weight_sum_support
    dict_to_check["weights_for_decoders"] = {}
    dict_to_check["weights_for_decoders"]["decoders"] = weight_for_decoder
    dict_to_check["weights_for_decoders"]["supporting"] = weight_for_support
    return dict_to_check


def get_recommend_base_num_features(num_output_classes):
    if isinstance(num_output_classes, (int, float)):
        return int(default_base_num_feature)

    if isinstance(num_output_classes, list):
        num_output_classes = len(num_output_classes)
        if num_output_classes >= 10:
            return default_base_num_feature
        else:
            return int(default_base_num_feature)

    return default_base_num_feature


def get_recommend_num_conv_per_stage(num_output_classes):
    # if isinstance(num_output_classes, (int, float)):
    #     return default_num_conv_per_stage
    # ret_num_conv_per_stage = copy.deepcopy(default_num_conv_per_stage)
    # if isinstance(num_output_classes, list):
    #     num_output_classes = len(num_output_classes)
    #     num_stages = int(min(5, np.rint(num_output_classes / 8)))
    #     for i in range(num_stages):
    #         ret_num_conv_per_stage[i % 5] += 1
    #
    # return ret_num_conv_per_stage
    return default_num_conv_per_stage


def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier):
    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], \
        "network can only be one of the following: \'3d_lowres\', \'3d_fullres\', \'3d_cascade_fullres\'"

    dataset_directory = join(preprocessing_output_dir, task)

    plans, plans_file = None, None
    if network == '2d':
        plans_file_pkl = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
        plans_file_json = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.json")

    else:
        plans_file_pkl = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")
        plans_file_json = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.json")

    # save all models in
    if network_trainer.endswith("DDP"):
        network_trainer = network_trainer[:-len("_DDP")]

    plans_file_result_json = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier, "plans.json")
    plans_file_result_ddp_json = join(network_training_output_dir, network, task, network_trainer + "_DDP__" + plans_identifier, "plans.json")

    try:
        plans = load_pickle(plans_file_pkl)
    except FileNotFoundError:
        if os.path.exists(plans_file_json):
            plans = convert_keys_to_int(load_json(plans_file_json))
            plans_file = plans_file_json
        elif os.path.exists(plans_file_result_json):
            plans = convert_keys_to_int(load_json(plans_file_result_json))
            plans_file = plans_file_result_json
        elif os.path.exists(plans_file_result_ddp_json):
            plans = convert_keys_to_int(load_json(plans_file_result_ddp_json))
            plans_file = plans_file_result_ddp_json
    if plans is None:
        raise ValueError("{} plan.json file loading error".format(task))
    possible_stages = list(plans['plans_per_stage'].keys())
    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    if network == "2d" or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)

    if "3d_lowres" in network.lower():
        batch_dice = False
        # print("Sample dice + CE loss")
    else:
        batch_dice = True
        # print("Batch dice + CE loss")
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage
