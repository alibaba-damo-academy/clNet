#   Author @Dazhou Guo
#   Data: 03.01.2023

import argparse
import socket

import torch

from batchgenerators.utilities.file_and_folder_operations import *

from clnet.paths import default_plans_identifier
from clnet.run.default_configuration import get_continual_decoding_ensemble_setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("continual_decoding_ensemble_setup", help="path of the json file.containing previously trained models for decoding ensemble trainer")
    parser.add_argument("--local-rank", default=0, type=int)
    args = parser.parse_args()
    plans_identifier = default_plans_identifier
    cfg_json = args.continual_decoding_ensemble_setup
    # local_rank = args.local_rank
    # Get local rank from the environment variable provided by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])

    trainer_class, task_dict, cl_network, cl_trainer = get_continual_decoding_ensemble_setup(cfg_json, plans_identifier, rank=local_rank, is_ddp=True)
    pretrained_network_continual_learning = None
    pth_continual_decoding_ensemble_setup_json_cleaned = cfg_json[:-len(".json")] + "_train_cleaned.json"
    task_dict_to_dump = task_dict.copy()
    task_dict_to_dump["clnet_network"] = cl_network
    task_dict_to_dump["clnet_trainer"] = cl_trainer
    with open(pth_continual_decoding_ensemble_setup_json_cleaned, "w") as f:
        json.dump(task_dict_to_dump, f, indent=4)
    del task_dict_to_dump

    for task in task_dict:
        plans_file = task_dict[task]["plans_file"]
        fold = task_dict[task]["fold"]
        output_folder_name = task_dict[task]["output_folder_name"]
        dataset_directory = task_dict[task]["dataset_directory"]
        batch_dice = task_dict[task]["batch_dice"]
        stage = task_dict[task]["stage"]
        save_npz = task_dict[task]["save_npz"]
        disable_saving = task_dict[task]["disable_saving"]
        val_disable_overwrite = task_dict[task]["val_disable_overwrite"]
        disable_postprocessing_on_folds = task_dict[task]["disable_postprocessing_on_folds"]
        decompress_data = task_dict[task]["decompress_data"]
        deterministic = task_dict[task]["deterministic"]
        run_mixed_precision = task_dict[task]["run_mixed_precision"]
        disable_val = task_dict[task]["disable_validation"]

        # Pipeline logic:
        # FIRST, we initialize the network using the "task_dict". The "trainer_class" will go through all the tasks.
        # SECOND, after initialization the network, we will train the associated decoding head(s) one task at a time.
        lr_and_epochs_decoder = task_dict[task]["model_training_setup"]["decoders"]
        if lr_and_epochs_decoder is None or len(lr_and_epochs_decoder) == 0 or "all" in lr_and_epochs_decoder:
            trainer = trainer_class(task_dict, task, "decoders", "all", pretrained_network_continual_learning,
                                    plans_file, fold, local_rank=local_rank, output_folder=output_folder_name,
                                    dataset_directory=dataset_directory, batch_dice=batch_dice,
                                    stage=stage, unpack_data=decompress_data,
                                    deterministic=deterministic, fp16=run_mixed_precision)
            if lr_and_epochs_decoder is None:
                if local_rank == 0:
                    trainer.print_to_log_file("Initializing: %s" % task)
            else:
                if local_rank == 0:
                    trainer.print_to_log_file("Training Decoder: %s-%s" % (task, "all"))
            val_folder = "validation_raw_" + task + "_all_decoders"
            trainer = train_and_validate(trainer, disable_saving, save_npz, val_folder,
                                         disable_postprocessing_on_folds, val_disable_overwrite, disable_val)
            # save the trained decoding heads (w. / w.o. general encoder) model to cpu.
            pretrained_network_continual_learning = trainer.network.to(torch.device("cpu"))
        else:
            for decoder in lr_and_epochs_decoder:
                trainer = trainer_class(task_dict, task, "decoders", decoder, pretrained_network_continual_learning,
                                        plans_file, fold, local_rank=local_rank, output_folder=output_folder_name,
                                        dataset_directory=dataset_directory, batch_dice=batch_dice,
                                        stage=stage, unpack_data=decompress_data,
                                        deterministic=deterministic, fp16=run_mixed_precision)
                if local_rank == 0:
                    trainer.print_to_log_file("Training Decoder: %s-%s" % (task, decoder))
                val_folder = "validation_raw_" + task + "_" + decoder + "_decoders"
                trainer = train_and_validate(trainer, disable_saving, save_npz, val_folder,
                                             disable_postprocessing_on_folds, val_disable_overwrite, disable_val)
                pretrained_network_continual_learning = trainer.network.to(torch.device("cpu"))

        # THIRD, after finish training the decoders, we will update the decoding heads using supporting head(s) if any.
        lr_and_epochs_supporting = task_dict[task]["model_training_setup"]["supporting"]
        if len(task_dict[task]["supporting"]) != 0:
            if lr_and_epochs_supporting is None or len(
                    lr_and_epochs_supporting) == 0 or "all" in lr_and_epochs_supporting:
                trainer = trainer_class(task_dict, task, "supporting", "all", pretrained_network_continual_learning,
                                        plans_file, fold, local_rank=local_rank, output_folder=output_folder_name,
                                        dataset_directory=dataset_directory, batch_dice=batch_dice,
                                        stage=stage, unpack_data=decompress_data,
                                        deterministic=deterministic, fp16=run_mixed_precision)
                if lr_and_epochs_supporting is None:
                    if local_rank == 0:
                        trainer.print_to_log_file("Initializing Supported Decoder: %s" % task)
                else:
                    if local_rank == 0:
                        trainer.print_to_log_file("Training Supported Decoder: %s-%s" % (task, "all"))
                val_folder = "validation_raw_" + task + "_all_supporting"
                trainer = train_and_validate(trainer, disable_saving, save_npz, val_folder,
                                             disable_postprocessing_on_folds, val_disable_overwrite, disable_val)
                pretrained_network_continual_learning = trainer.network.to(torch.device("cpu"))
            else:
                for decoder in lr_and_epochs_supporting:
                    trainer = trainer_class(task_dict, task, "supporting", decoder,
                                            pretrained_network_continual_learning,
                                            plans_file, fold, local_rank=local_rank, output_folder=output_folder_name,
                                            dataset_directory=dataset_directory, batch_dice=batch_dice,
                                            stage=stage, unpack_data=decompress_data,
                                            deterministic=deterministic, fp16=run_mixed_precision)
                    if local_rank == 0:
                        trainer.print_to_log_file("Training Supported Decoder: %s-%s" % (task, decoder))
                    val_folder = "validation_raw_" + task + "_" + decoder + "_supporting"
                    trainer = train_and_validate(trainer, disable_saving, save_npz, val_folder,
                                                 disable_postprocessing_on_folds, val_disable_overwrite, disable_val)
                    pretrained_network_continual_learning = trainer.network.to(torch.device("cpu"))


def train_and_validate(trainer, disable_saving, npz, val_folder, disable_postprocessing_on_folds,
                       val_disable_overwrite, disable_val):
    if disable_saving:
        trainer.save_final_checkpoint = False
        trainer.save_best_checkpoint = False
        trainer.save_intermediate_checkpoints = True
        trainer.save_latest_only = True

    skip_training = False
    load_from_ema = True
    continual_training_states = trainer.continue_training_check()

    trainer.initialize()

    # We assume the trained "task sequence" is fixed.
    # If the "current task" is before "continue task" (i.e. "current task" already trained), then we skip the training.
    if continual_training_states is not None:
        if continual_training_states['trained_head']:
            skip_training = True
        elif continual_training_states['same_head']:
            load_from_ema = False

    if trainer.initial_lr is None or trainer.max_num_epochs is None or trainer.task_pretrain_model_name is not None:
        # We try to load weights from the EMA module if the "head" is already trained.
        # Otherwise, we do not load weights from the EAM module.
        trainer.load_pretrained_params_ensemble(load_from_ema=load_from_ema, is_training=not skip_training)

    if not skip_training:
        # When "skip_training" is False + "continue_training" is True,
        # Based on the "skip_training" condition, it will skip training the following heads of the current task
        # Solution: Once "skip_training" is False, we set the "current task" "continue_training" to False,
        # s.t., the following head can be also trained.
        # trainer.train_dict["continue_training"] = False
        if "load_only_encoder" in trainer.train_dict and trainer.train_dict["load_only_encoder"]:
            return trainer
        else:
            trainer.run_training()
            trainer.network.eval()
            if not disable_val:
                trainer.validate(save_softmax=npz, validation_folder_name=val_folder,
                                 run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                                 overwrite=not val_disable_overwrite, all_in_gpu=True)
    return trainer


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ["OMP_NUM_THREADS"] = "1"
    if 'MASTER_PORT' not in os.environ.keys():
        port = str(find_free_network_port())
        print(f"using port {port}")
        os.environ['MASTER_PORT'] = port
    main()
