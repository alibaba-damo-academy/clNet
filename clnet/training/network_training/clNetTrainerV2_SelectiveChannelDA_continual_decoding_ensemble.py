#   Author @Dazhou Guo
#   Data: 03.01.2023

import torch
import shutil
import numpy as np
import matplotlib
from time import sleep
from multiprocessing import Pool

from clnet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from clnet.evaluation.evaluator import aggregate_scores
from clnet.postprocessing.connected_components import determine_postprocessing
from clnet.training.network_training.clNetTrainer import clNetTrainer
from clnet.training.dataloading.dataset_loading import unpack_dataset
from clnet.training.loss_functions.deep_supervision import MultipleOutputLossEnsemble
from clnet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from clnet.configuration import default_num_threads

from batchgenerators.utilities.file_and_folder_operations import *

matplotlib.use("agg")


class clNetTrainerV2_SelectiveChannelDA_Continual_Decoding_Ensemble(clNetTrainer):
    """
    General continual learning training template for adding or updating decoder heads
    """
    def __init__(self, task_dict, task, decoder_or_support, head_to_train, pretrained_network_continual_learning,
                 plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None, unpack_data=True, deterministic=True, fp16=True):
        super().__init__(task_dict, task, decoder_or_support, head_to_train, pretrained_network_continual_learning,
                         plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

    def process_plans(self, plans):
        super().process_plans(plans)
        self.set_batch_size_and_oversample()

    def initialize(self, predetermined_network=None, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or self.plans is None:
                self.load_plans_file()
            self.process_plans(self.plans)

            # self.normalization_schemes = self.plans['normalization_schemes']
            if "model_training_setup" in self.train_dict and "patch_size" in self.train_dict["model_training_setup"]:
                if self.train_dict["model_training_setup"]["patch_size"][self.head_to_train] is not None:
                    self.patch_size = self.train_dict["model_training_setup"]["patch_size"][self.head_to_train]

            self.setup_data_aug_params()

        self.was_initialized = True

    def set_batch_size_and_oversample(self):
        default_batches = self.batch_size
        self.batch_size = self.train_dict["model_training_setup"]["batch_size"][self.head_to_train]

        batch_per_epoch_ratio = max(1, np.round(self.batch_size / default_batches).astype(int))
        self.num_train_batches_per_epoch = max(1, np.round(self.num_train_batches_per_epoch / batch_per_epoch_ratio).astype(int))
        self.num_val_batches_per_epoch = max(1, np.round(self.num_val_batches_per_epoch / batch_per_epoch_ratio).astype(int))

    def load_pretrained_params_ensemble(self, load_from_ema=False, is_training=True):
        super()._load_pretrained_params_ensemble(load_from_ema, is_training)
        if torch.cuda.is_available():
            if is_training:
                self.initialize_optimizer_and_scheduler()
