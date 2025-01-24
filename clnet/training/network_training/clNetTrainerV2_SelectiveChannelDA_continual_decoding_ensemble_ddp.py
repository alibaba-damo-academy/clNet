#   Author @Dazhou Guo
#   Data: 03.01.2023


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
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            self.loss = MultipleOutputLossEnsemble(self.task_classes, self.batch_dice, self.ds_loss_weights, True)

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans["data_identifier"] + "_stage%d" % self.stage)
            if training and self.initial_lr is not None and self.max_num_epochs is not None:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    if self.local_rank == 0:
                        print("unpacking dataset")
                        unpack_dataset(self.folder_with_preprocessed_data)
                        print("done")
                    distributed.barrier()
                else:
                    if self.local_rank == 0:
                        print("INFO: Not unpacking data! Training may be slow due to that."
                              "Pray you are not using 2d or you will wait all winter for your model to finish!")

                seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get("num_threads"))
                seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get("num_threads") // 2, 1))
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val, self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params, deep_supervision_scales=self.deep_supervision_scales,
                    seeds_train=seeds_train, seeds_val=seeds_val, pin_memory=self.pin_memory)

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())), also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())), also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            network_requires_grad = self.initialize_optimizer_and_scheduler()
            self.compile_network()
            self.initialize_prune_ratio()
            if self.max_num_epochs is not None and isinstance(self.max_num_epochs, (int, float)):
                self.print_to_log_file("Initial Learning: {}".format(self.initial_lr))
            if self.initial_lr is not None and isinstance(self.initial_lr, (int, float)):
                self.print_to_log_file("Maximum Training Epochs: {}".format(self.max_num_epochs))
            self.initialize_ddp(network_requires_grad)
        else:
            self.print_to_log_file("self.was_initialized is True, not running self.initialize again")
        self.was_initialized = True
        if self.initial_lr is not None:
            self.print_to_log_file("%s - %s - %s  - Batch Size: %d" % (self.task, self.head_to_train, self.decoder_or_support, self.batch_size))
            self.print_to_log_file("%s - %s - %s  - Patch Size: [%d, %d, %d]" % (self.task, self.head_to_train, self.decoder_or_support,
                                                                                 self.patch_size[0], self.patch_size[1], self.patch_size[2]))

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

    def validate(self, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = self.network.do_ds
        net.set_do_ds(False)  # ! set do_ds=False for all decoders

        do_mirroring = not self.train_dict["no_mirroring"]
        if do_mirroring:
            print("WARNING! do_mirroring was True but we cannot do that because we trained without mirroring. "
                  "do_mirroring was set to False")

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        results = []

        all_keys = list(self.dataset_val.keys())
        my_keys = all_keys[self.local_rank::dist.get_world_size()]

        for k in my_keys:
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"), join(self.gt_niftis_folder, fname + ".nii.gz")])
            if k in my_keys:
                if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                        (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                    data = np.load(self.dataset[k]['data_file'])['data']

                    data[-1][data[-1] == -1] = 0
                    softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax_ensemble(
                        data[:-1], self.decoder_or_support, self.head_to_train, do_mirroring=do_mirroring,
                        mirror_axes=mirror_axes, use_sliding_window=use_sliding_window, step_size=step_size,
                        use_gaussian=use_gaussian, all_in_gpu=all_in_gpu, mixed_precision=self.fp16)[1]
                    softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be saved
                        np.save(join(output_folder, fname + ".npy"), softmax_pred)
                        softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                             ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                               properties, interpolation_order, self.regions_class_order,
                                                               None, None, softmax_fname, None, force_separate_z,
                                                               interpolation_order_z),)))

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")
        distributed.barrier()

        if self.local_rank == 0:
            # evaluate raw predictions
            self.print_to_log_file("evaluation of raw predictions")
            task = self.dataset_directory.split("/")[-1]
            job_name = self.experiment_name
            _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                                 json_output_file=join(output_folder, "summary.json"),
                                 json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                                 json_author="Fabian",
                                 json_task=task, num_threads=self.default_num_threads)

            if run_postprocessing_on_folds:
                self.print_to_log_file("determining postprocessing")
                determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                         final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
                # after this the final predictions for the vlaidation set can be
                # found in validation_folder_name_base + "_postprocessed"
                # They are always in that folder, even if no postprocessing as applied!

            # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
            # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
            # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
            # be used later
            gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
            maybe_mkdir_p(gt_nifti_folder)
            for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
                success = False
                attempts = 0
                e = None
                while not success and attempts < 10:
                    try:
                        shutil.copy(f, gt_nifti_folder)
                        success = True
                    except OSError as e:
                        attempts += 1
                        sleep(1)
                if not success:
                    print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                    if e is not None:
                        raise e

        self.network.train(current_mode)
        net.set_to_eval_mode(self.train_dict, self.decoder_or_support, self.head_to_train)
        net.set_do_ds(ds)  # ! recover do_ds

    def plot_progress(self):
        if self.local_rank == 0:
            self._plot_progress()

    def save_checkpoint(self, fname, save_optimizer=True):
        if self.local_rank == 0:
            self._save_checkpoint(fname, save_optimizer)

    def run_training(self):
        if self.train_dict["plot_network"]:
            super()._plot_network_architecture()

        if self.max_num_epochs is not None and self.initial_lr is not None:
            if self.local_rank == 0:
                self.save_debug_information()
            super()._run_training()
            # unwrap the DDP -> for validation()
            self.network = self.network.module
