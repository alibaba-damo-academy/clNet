import shutil
import numpy as np
from multiprocessing import Pool

from time import sleep

import torch
import torch.distributed as dist
from torch import nn, distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from clnet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from clnet.evaluation.evaluator import aggregate_scores
from clnet.postprocessing.connected_components import determine_postprocessing
from clnet.training.loss_functions.deep_supervision import MultipleOutputLossEnsemble
from clnet.training.network_training.clNetTrainer import clNetTrainer
from clnet.training.dataloading.dataset_loading import unpack_dataset
from clnet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from clnet.configuration import *

from batchgenerators.utilities.file_and_folder_operations import *


class clNetTrainerV2_SelectiveChannelDA_Continual_Decoding_Ensemble_DDP(clNetTrainer):
    """
    General continual learning training template for adding or updating decoder heads
    """
    def __init__(self, task_dict, task, decoder_or_support, head_to_train, pretrained_network_continual_learning,
                 plans_file, fold, local_rank=0, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None, unpack_data=True, deterministic=True, fp16=True):
        super().__init__(task_dict, task, decoder_or_support, head_to_train, pretrained_network_continual_learning,
                         plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        np.random.seed(local_rank)
        torch.manual_seed(local_rank)
        self.is_ddp = dist.is_available()
        # self.local_rank = local_rank
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_batch_size = None

        if torch.cuda.is_available() and self.is_ddp:
            torch.cuda.manual_seed_all(local_rank)
            torch.cuda.set_device(local_rank)

        if not torch.distributed.is_initialized():
            self.setup_ddp()
            # dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=7000))

    def setup_ddp(self):
        """Initialize the process group for distributed training."""
        # Get environment variables set by torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize the distributed process group (NCCL backend for GPU)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        # Set the current device based on the local rank
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank

    def set_batch_size_and_oversample(self):
        # Setup DDP over-sampling rate
        batch_sizes = []
        oversample_percents = []

        world_size = dist.get_world_size()
        my_rank = dist.get_rank()

        default_batches = self.batch_size
        self.batch_size = self.train_dict["model_training_setup"]["batch_size"][self.head_to_train]
        num_total_threads = multiprocessing.cpu_count()
        self.batch_size = max(2, np.round(self.batch_size / world_size).astype(int))
        self.global_batch_size = self.batch_size * world_size

        batch_per_epoch_ratio = max(1, np.round(self.global_batch_size / default_batches).astype(int))
        self.default_num_threads = min(num_total_threads, batch_per_epoch_ratio * self.default_num_threads)

        self.num_train_batches_per_epoch = max(1, np.round(self.num_train_batches_per_epoch / batch_per_epoch_ratio).astype(int))

        for rank in range(world_size):
            batch_sizes.append(self.batch_size)

            sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
            sample_id_high = np.sum(batch_sizes)
            if sample_id_high / self.global_batch_size < (1 - self.ofp):
                oversample_percents.append(0.0)
            elif sample_id_low / self.global_batch_size > (1 - self.ofp):
                oversample_percents.append(1.0)
            else:
                percent_covered_by_this_rank = sample_id_high / self.global_batch_size - sample_id_low / self.global_batch_size
                oversample_percent_here = 1 - (((1 - self.ofp) - sample_id_low / self.global_batch_size) / percent_covered_by_this_rank)
                oversample_percents.append(oversample_percent_here)

        if self.local_rank == 0:
            self.print_to_log_file("Using threads: ", self.default_num_threads)
            self.print_to_log_file("Total batch size: ", self.global_batch_size)
            self.print_to_log_file("Batch size per GPU: ", batch_sizes)
            self.print_to_log_file("Number of batch per training epoch: ", self.num_train_batches_per_epoch, "(", batch_per_epoch_ratio, ")")

        self.batch_size = batch_sizes[my_rank]
        self.rank_batch_size = batch_sizes
        self.ofp = oversample_percents[my_rank]
        if self.initial_lr is not None:
            self.initial_lr *= len(batch_sizes)
        if self.weight_decay is not None:
            self.weight_decay *= max(1, int(len(batch_sizes) / 2))

    def set_oversample_rate(self):
        if isinstance(self.network, DDP):
            oversample_percents = []
            batch_sizes = []
            world_size = dist.get_world_size()
            my_rank = dist.get_rank()
            batch_size_before_ddp = np.sum(self.rank_batch_size)
            batch_size_per_GPU = np.ceil(batch_size_before_ddp / world_size).astype(int)

            for rank in range(world_size):
                batch_sizes.append(batch_size_per_GPU)

                sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
                sample_id_high = np.sum(batch_sizes)
                if sample_id_high / self.global_batch_size < (1 - self.ofp):
                    oversample_percents.append(0.0)
                elif sample_id_low / self.global_batch_size > (1 - self.ofp):
                    oversample_percents.append(1.0)
                else:
                    percent_covered_by_this_rank = sample_id_high / self.global_batch_size - sample_id_low / self.global_batch_size
                    oversample_percent_here = 1 - (((1 - self.ofp) - sample_id_low / self.global_batch_size) / percent_covered_by_this_rank)
                    oversample_percents.append(oversample_percent_here)

            self.ofp = oversample_percents[my_rank]

    def process_plans(self, plans):
        super().process_plans(plans)
        self.set_batch_size_and_oversample()

    def initialize(self, predetermined_network=None, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or self.plans is None:
                self.load_plans_file()
            self.process_plans(self.plans)

            # self.normalization_schemes = self.plans["normalization_schemes"]
            if "model_training_setup" in self.train_dict and "patch_size" in self.train_dict["model_training_setup"]:
                if self.train_dict["model_training_setup"]["patch_size"][self.head_to_train] is not None:
                    self.patch_size = self.train_dict["model_training_setup"]["patch_size"][self.head_to_train]

            self.setup_data_aug_params()
            self.initialize_network()

    def initialize_ddp(self, network_requires_grad):
        if isinstance(self.network, DDP):
            # check if the network is already wrapped in DDP -> prevent nested wrapping
            self.network = self.network.module
        if network_requires_grad:
            self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)

    def load_pretrained_params_ensemble(self, load_from_ema=False, is_training=True):
        super()._load_pretrained_params_ensemble(load_from_ema, is_training)
        if torch.cuda.is_available():
            if is_training:
                self.initialize_optimizer_and_scheduler()