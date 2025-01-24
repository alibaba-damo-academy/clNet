import os
import copy
import json

from clnet.configuration import *
from clnet.paths import default_plans_identifier
from clnet.run.default_configuration import get_continual_decoding_ensemble_setup


def cfg_parser_for_inference(clnet_cfg: str, plans_identifier: str = default_plans_identifier, task_plan_to_use: str = None):
    trainer_class, task_dict, _, _ = \
        get_continual_decoding_ensemble_setup(clnet_cfg, plans_identifier, load_plan_from_pretrain=True, inference=True)

    try_to_find_models = ["model_final_checkpoint.model", "model_best.model", "model_latest.model"]
    try_to_find_folds = ["all"] + ["fold_" + str(i) for i in range(10)]
    trainer_heads_details = {}
    # Here we set the "default_training_setup" loading to be True.
    # Try to locate each task defined in cfg.json file.
    for task in task_dict:
        pth_pretrained_model = None
        flag_pretrained_model_exists = False
        if task_plan_to_use is not None and task_dict[task]["task"] == task_plan_to_use:
            task_plan_to_use = task
        # We set the "continue_training" to False
        task_dict[task]["continue_training"] = False
        # 1st, try to check the predefined model path. If exists, then use it.
        # If not, we try to search all possible paths
        if task_dict[task]["pretrain_model_name"] is not None:
            if task_dict[task]["fold"] == "all":
                pth_pretrained_model = \
                    os.path.join(task_dict[task]["output_folder_name"], task_dict[task]["fold"], task_dict[task]["pretrain_model_name"])
            elif isinstance(task_dict[task]["fold"], int):
                pth_pretrained_model = \
                    os.path.join(task_dict[task]["output_folder_name"], "fold_" + str(task_dict[task]["fold"]), task_dict[task]["pretrain_model_name"])
        # 2nd, try to locate possible pretrained model.
        if pth_pretrained_model is not None and os.path.exists(pth_pretrained_model):
            flag_pretrained_model_exists = True
        else:
            current_fold = "fold_" + str(task_dict[task]["fold"])
            if current_fold in try_to_find_folds:
                try_to_find_folds.remove(current_fold)
                try_to_find_folds = [current_fold] + try_to_find_folds
            for try_to_find_fold in try_to_find_folds:
                for try_to_find_model in try_to_find_models:
                    pth_pretrained_model = os.path.join(task_dict[task]["output_folder_name"], try_to_find_fold, task + "_" + try_to_find_model)
                    if os.path.exists(pth_pretrained_model):
                        flag_pretrained_model_exists = True
                        task_dict[task]["pretrain_model_name"] = task + "_" + try_to_find_model
                        if try_to_find_fold != "all":
                            task_dict[task]["fold"] = int(try_to_find_fold[len("fold_"):])
                        else:
                            task_dict[task]["fold"] = try_to_find_fold
                        break
                    pth_pretrained_model = os.path.join(task_dict[task]["output_folder_name"], try_to_find_fold, try_to_find_model)
                    if os.path.exists(pth_pretrained_model):
                        flag_pretrained_model_exists = True
                        task_dict[task]["pretrain_model_name"] = try_to_find_model
                        if try_to_find_fold != "all":
                            task_dict[task]["fold"] = int(try_to_find_fold[len("fold_"):])
                        else:
                            task_dict[task]["fold"] = try_to_find_fold
                        break
                if flag_pretrained_model_exists:
                    break
        if flag_pretrained_model_exists:
            trainer_heads_details[task] = task_dict[task]
    # Dump the "cleaned" prediction JSON
    pth_continual_decoding_ensemble_pred_json_cleaned = clnet_cfg[:-len(".json")] + "_pred_cleaned.json"

    with open(pth_continual_decoding_ensemble_pred_json_cleaned, "w") as f:
        json.dump(task_dict, f, indent=4)

    # Get body-part scores for each decoding head
    trainer_heads_summarized = {"decoders": {}, "supporting": {}, "bpr_range": {}, "patch_size": {}}
    for task in trainer_heads_details:
        current_decoder = task_dict[task]["decoders"]
        current_support = task_dict[task]["supporting"]
        current_patch_size = task_dict[task]["model_training_setup"]["patch_size"]
        # 1st, initialize the patch size using the "default_patch_size" defined in paths.py
        tmp_patch_size = None
        # 2nd, find if there is "all" decoding heads share the same patch size
        for head in current_patch_size:
            if head == "all":
                if isinstance(current_patch_size[head], list) and len(current_patch_size[head]) == 3:
                    tmp_patch_size = current_patch_size[head]
                    break
        # 3rd, parse all decoding heads
        if len(current_decoder) != 0:
            for head in current_decoder:
                trainer_heads_summarized["decoders"][head] = {}
                trainer_heads_summarized["decoders"][head]['task'] = task
                trainer_heads_summarized["decoders"][head]['labels'] = task_dict[task]["decoders"][head]
                trainer_heads_summarized["patch_size"][head] = tmp_patch_size
        # 4th, parse all supporting heads
        if len(current_support) != 0:
            for head in current_support:
                trainer_heads_summarized["supporting"][head] = {}
                # If we found the same head as we determined in the "decoders",
                # we will delete it and use only "supporting" for prediction.
                # if head in trainer_heads_summarized["decoders"]:
                #     del trainer_heads_summarized["decoders"][head]
                trainer_heads_summarized["supporting"][head]['task'] = task
                trainer_heads_summarized["supporting"][head]['labels'] = task_dict[task]["decoders"][head]
                trainer_heads_summarized["supporting"][head]['supporting'] = task_dict[task]["supporting"][head]
                if head not in trainer_heads_summarized["patch_size"]:
                    trainer_heads_summarized["patch_size"][head] = tmp_patch_size
        # 5th, parse patch sizes
        for head in current_patch_size:
            tmp_patch_size = current_patch_size[head]
            if isinstance(tmp_patch_size, list) and len(tmp_patch_size) == 3 and head != "all":
                trainer_heads_summarized["patch_size"][head] = tmp_patch_size
        trainer_heads_summarized['bpr_range'].update(task_dict[task]["bpr_range_for_decoders"])

    return trainer_class, trainer_heads_summarized, trainer_heads_details


def cfg_parser_for_device(task_dict, summarized_dict, heads_to_pred, decoder_or_support):
    task_dict_to_pred = {}
    # The training "dict" contains multiple "tasks".
    # Try to copy the "tasks" that include the head in "heads_to_pred" list
    for task in task_dict:
        if task_dict[task]["type"] == "GeneralEncoder":
            task_dict_to_pred[task] = copy.deepcopy(task_dict[task])
        else:
            if decoder_or_support == "decoders":
                for head in heads_to_pred:
                    if head in task_dict[task][decoder_or_support]:
                        # we will use the latest task to load the weights of the head
                        task_dict_to_pred[task] = copy.deepcopy(task_dict[task])
            else:
                for head in heads_to_pred:
                    if head in task_dict[task][decoder_or_support]:
                        task_dict_to_pred[task] = copy.deepcopy(task_dict[task])
                        # We also need to copy the tasks that include the supporting decoders.
                        # If the head is in summarized_dict["supporting"], then we will try to find all decoding tasks that support this head
                        if head in summarized_dict[decoder_or_support]:
                            for supporting_head in summarized_dict[decoder_or_support]:
                                # Try to find the supporting tasks
                                if supporting_head in summarized_dict["decoders"] and \
                                        summarized_dict[decoder_or_support][supporting_head]["task"] in task_dict:
                                    task_dict_to_pred[summarized_dict[decoder_or_support][supporting_head]["task"]] = \
                                        copy.deepcopy(task_dict[summarized_dict[decoder_or_support][supporting_head]["task"]])

    # Each "task" contains multiple "heads".
    # Aim to correctly enlist the "heads" in the "heads_to_pred" list
    summarized_dict_cleaned = copy.deepcopy(summarized_dict)
    heads_to_pred_cleaned = []
    for head in summarized_dict[decoder_or_support]:
        if head not in heads_to_pred:
            del summarized_dict_cleaned[decoder_or_support][head]
    for task in task_dict_to_pred:
        task_dict_to_pred[task]["decoders"] = {}
        task_dict_to_pred[task]["supporting"] = {}

    for task in task_dict_to_pred:
        if task_dict_to_pred[task]["type"] == "GeneralEncoder":
            if task_dict_to_pred[task]["load_only_encoder"]:
                continue
        for head in task_dict[task][decoder_or_support]:
            if decoder_or_support == "decoders":
                if head in summarized_dict_cleaned["decoders"] and head not in summarized_dict_cleaned["supporting"]:
                    task_dict_to_pred[task][decoder_or_support][head] = copy.deepcopy(task_dict[task][decoder_or_support][head])
                    # task_dict_to_pred[task]["decoder_architecture_setup"][head] = copy.deepcopy(task_dict[task]["decoder_architecture_setup"][head])
                    heads_to_pred_cleaned.append(head)
            else:
                if head in summarized_dict_cleaned["supporting"]:
                    task_dict_to_pred[task]["supporting"][head] = copy.deepcopy(task_dict[task]["supporting"][head])
                    task_dict_to_pred[task]["decoders"][head] = copy.deepcopy(task_dict[task]["decoders"][head])
                    # task_dict_to_pred[task]["decoder_architecture_setup"][head] = copy.deepcopy(task_dict[task]["decoder_architecture_setup"][head])
                    heads_to_pred_cleaned.append(head)
    if decoder_or_support == "supporting":
        for head in summarized_dict_cleaned["supporting"]:
            task = summarized_dict_cleaned["supporting"][head]["task"]
            task_dict_to_pred[task]["supporting"][head] = copy.deepcopy(task_dict[task]["supporting"][head])
            task_dict_to_pred[task]["decoders"][head] = copy.deepcopy(task_dict[task]["decoders"][head])
            # task_dict_to_pred[task]["decoder_architecture_setup"][head] = copy.deepcopy(task_dict[task]["decoder_architecture_setup"][head])
            for supporting_head in summarized_dict_cleaned["supporting"][head]["supporting"]:
                supporting_task = summarized_dict_cleaned["decoders"][supporting_head]["task"]
                task_dict_to_pred[supporting_task]["decoders"][supporting_head] = \
                    copy.deepcopy(task_dict[supporting_task]["decoders"][supporting_head])

    task_dict_to_pred_cleaned = {}
    for task in task_dict_to_pred:
        if task_dict[task]["type"] == "GeneralEncoder":
            task_dict_to_pred_cleaned[task] = copy.deepcopy(task_dict_to_pred[task])
        else:
            for head in heads_to_pred_cleaned:
                if head in task_dict_to_pred[task]["decoders"] or head in task_dict_to_pred[task]["supporting"]:
                    task_dict_to_pred_cleaned[task] = copy.deepcopy(task_dict_to_pred[task])
                    break

    return task_dict_to_pred_cleaned, heads_to_pred_cleaned


if __name__ == "__main__":
    # clnet_cfg = "/nas/dazhou.guo/Projects/clNet/clnet/training_cfg_json/CSS_PL_ALL.json"
    # heads_to_pred = ["decoder1", "decoder2", "decoder3", "decoder10", "decoder7"]
    clnet_cfg = "/nas/dazhou.guo/Projects/clNet/clnet/training_cfg_json/CSS_StructSeg_GEv1.json"
    heads_to_pred = ["StructSeg_V1"]
    trainer_class, trainer_heads_summarized, trainer_heads_details = cfg_parser_for_inference(clnet_cfg, default_plans_identifier, None)

    task_dict_for_device_decoder = cfg_parser_for_device(trainer_heads_details, trainer_heads_summarized, heads_to_pred, "decoders")
    a = 1
    task_dict_for_device_support = cfg_parser_for_device(trainer_heads_details, trainer_heads_summarized, heads_to_pred, "supporting")
    a = 1
