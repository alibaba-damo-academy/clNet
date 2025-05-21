import sys
import copy

from typing import Tuple
from warnings import warn
import matplotlib
import matplotlib.pyplot as plt
from abc import abstractmethod
from collections import OrderedDict

from time import time
from time import sleep
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold

try:
    from torch._dynamo import OptimizedModule
except ModuleNotFoundError as e:
    # This exception is raised if the `torch._dynamo` module does not exist.
    print(f"ModuleNotFoundError: {e}. It seems `torch._dynamo` is not available in your PyTorch installation.")
    OptimizedModule = None
except ImportError as e:
    # This exception is raised if `OptimizedModule` is not found within the `torch._dynamo` module.
    print(f"ImportError: {e}. The `OptimizedModule` class might not be present in `torch._dynamo`.")
    OptimizedModule = None
except Exception as e:
    # Catches any other unexpected exceptions.
    print(f"An unexpected exception occurred: {e}")
    OptimizedModule = None

import torch.distributed as dist
import torch.amp
from torch.amp import autocast
import torch.backends.cudnn as cudnn
from torch.nn.functional import interpolate
from torch.nn.parallel import DistributedDataParallel as DDP

import clnet
from clnet.configuration import *
from clnet.sparse_conv.SparseConvWraper import *
from clnet.utilities.nd_softmax import softmax_helper
from clnet.utilities.distributed import AllGatherGrad
from clnet.utilities.tensor_utilities import sum_tensor
from clnet.utilities.to_torch import maybe_to_torch, to_cuda
from clnet.training.dataloading.dataset_loading import load_dataset
from clnet.network_architecture.initialization import InitWeights_He
from clnet.training.dataloading.dataset_loading import DataLoader3D, DataLoader2D
from clnet.training.learning_rate.poly_lr import combined_lr_lambda
from clnet.network_architecture.generic_UNet_continual import Generic_UNet_Continual_Base
from clnet.training.loss_functions.dice_loss import RobustCrossEntropyLoss
from clnet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, get_patch_size, default_3D_augmentation_params
from clnet.network_architecture.custom_modules.pruning_modules import perform_network_initialization_with_pruning_capability

from batchgenerators.utilities.file_and_folder_operations import *

matplotlib.use("agg")
torch.backends.cudnn.benchmark = True


class clNetTrainer(object):
    def __init__(self, task_dict, task, decoder_or_support, head_to_train, pretrained_network_continual_learning,
                 plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=True):

        self.fp16 = fp16
        self.amp_grad_scaler = None

        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        self.network = self.initial_lr = self.net_pool_per_axis = self.data_aug_params = self.transpose_forward = self.transpose_backward = self.dataset = \
            self.max_num_epochs = self.plans = self.tr_gen = self.val_gen = self.lr_scheduler = self.optimizer = self.dataset_tr = self.dataset_val = \
            self.regions_class_order = self.loss = self.deep_supervision_scales = self.ds_loss_weights = self.prune_start_epoch = self.rank_batch_size = \
            self.classes = self.do_dummy_2D_aug = self.use_mask_for_norm = self.only_keep_largest_connected_component = self.pad_all_sides = self.log_file = \
            self.dl_tr = self.dl_val = self.min_region_size_per_class = self.min_size_per_classNone = self.continual_training_states = \
            self.folder_with_preprocessed_data = None

        self.unpack_data = unpack_data
        self.epoch = 0
        self.prune_repeat_flag = False
        self.local_rank = 0
        self.also_val_in_tr_mode = False
        # time estimation
        self.time_data_loading = []
        self.time_data_target = []
        self.time_data_to_cuda = []
        self.time_train_forward = []
        self.time_train_backward = []
        self.time_loss = []
        # time estimation
        self.stage = stage
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder
        self.fold = fold
        self.batch_dice = batch_dice
        self.update_fold(fold)
        self.weight_decay = default_weight_decay

        if self.dataset_directory is not None and isdir(self.dataset_directory):
            self.gt_niftis_folder = join(self.dataset_directory, "gt_segmentations")
        else:
            self.gt_niftis_folder = None

        ################# SET THESE IN self.initialize() ###################################
        self.was_initialized = False

        ################# LEAVE THESE ALONE ################################################
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.deterministic = deterministic

        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

        # loaded automatically from plans_file
        self.num_input_channels = 1

        self.base_num_features = default_base_num_feature
        self.net_num_pool_op_kernel_sizes = default_pool
        self.patch_size = default_patch_size
        self.batch_size = default_batch_size
        self.conv_per_stage = default_num_conv_per_stage

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        self.use_progress_bar = False
        if 'clnet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['clnet_use_progress_bar']))
        self.task_dict = task_dict
        self.unpack_data = unpack_data
        self.train_dict = task_dict[task]
        self.task = task
        self.task_classes = []
        self.task_pretrain_model_name = task_dict[task]["pretrain_model_name"]
        self.decoder_or_support = decoder_or_support
        self.head_to_train = head_to_train
        self.load_pretrained_decoder = False

        self.num_train_batches_per_epoch = default_num_train_batches_per_epoch
        self.num_val_batches_per_epoch = default_num_val_batches_per_epoch
        self.save_every = default_save_every
        self.default_num_threads = default_num_threads
        self.backup_num_batches_per_epoch, self.backup_num_val_batches_per_epoch = 0, 0

        self.ce_loss = RobustCrossEntropyLoss()

        self.ofp = 0.33
        self.pin_memory = True
        #
        self.pretrained_network_continual_learning = pretrained_network_continual_learning
        self.online_eval_foreground_dsc_pruning = {}

        self.online_eval_tp = {}
        self.online_eval_fp = {}
        self.online_eval_fn = {}
        self.all_val_eval_metrics = {}

        self.online_eval_tp_ema = {}
        self.online_eval_fp_ema = {}
        self.online_eval_fn_ema = {}
        self.all_val_eval_metrics_ema = {}

        # default is set True to all decoders.
        # If the mean DSC drop is more than default_threshold, then the decoder is done pruning.
        # Please note that self.prune_if_is_done is reset for each training task listed in the "train_cfg.json" file.
        # This will make sure that the same decoding head from downstream tasks could be further pruned.
        self.prune_if_is_done = {}
        # Check the "pruning mask" for each decoder
        self.prune_decoder_sparsity_before_prune = {}
        # If the decoder requires pruning
        self.prune_if_to_perform = {}
        self.state_dict_unpruned = {}
        # The initial state dict before pruning
        # |--> lottery-ticket pruning -- use the state_dict weights before each prune to re-initialize the weight.
        self.state_dict_lth_reinit_prune = {}
        # save the previous state of the decoder --> restore the decoder if the performance drop exceeds the threshold.
        self.state_dict_before_each_prune = {}
        # the final pruning ratio of each head
        self.prune_ratios_final_for_each_decoder = {}
        self.run_epoch_before_prune = False
        self.hook_only_once = True
        # the pruning ratio to try --> will be adapted based on the current decoding head architecture and sparsity
        self.prune_ratios_to_try = {}
        self.prune_repeat_times = {}
        self.prune_ratio_in_percentage = {}
        self.prune_all_val_eval_metrics_moving_average = {}

        self.flag_convert_to_sparse = {}

        if decoder_or_support != "load_all":
            model_training_setup = self.train_dict["model_training_setup"][decoder_or_support]
            if model_training_setup is not None and head_to_train in model_training_setup and len(model_training_setup[head_to_train]) == 7:
                # oversample_foreground_percent controls the percentage of training samples that must have foreground,
                # first value is initial percentage at epoch 0, second is final percentage at max_num_epoch
                self.oversample_foreground_percent = model_training_setup[head_to_train][:2]
                self.oversample_foreground_percent[0] = float(self.oversample_foreground_percent[0])
                self.oversample_foreground_percent[1] = float(self.oversample_foreground_percent[1])
                self.initial_lr, self.max_num_epochs, self.load_pretrained_decoder, self.perform_pruning, _ = model_training_setup[head_to_train][2:]
                self.ofp = self.oversample_foreground_percent[0]

        if self.max_num_epochs is not None:
            # If the decoder requires pruning
            # Then, we add additional training epochs to perform pruning and performance recovery
            if self.train_dict["model_training_setup"][self.decoder_or_support][head_to_train][-2]:
                self.prune_start_epoch = self.max_num_epochs
                self.max_num_epochs = default_pruning_performance_recovery_epochs + len(default_pruning_percentages_to_try) + self.max_num_epochs
                self.train_dict["model_training_setup"][self.decoder_or_support][head_to_train][3] = self.max_num_epochs

        task_foreground_classes = []
        if self.head_to_train == "all":
            if self.decoder_or_support != "load_all":
                for head in self.train_dict[self.decoder_or_support]:
                    fg_cls = self.train_dict[self.decoder_or_support][head]
                    fg_cls = [fg_cls] if not isinstance(fg_cls, list) else fg_cls
                    task_foreground_classes += fg_cls
                    self.prune_repeat_times[head] = default_pruning_repeat_times
        else:
            fg_cls = self.train_dict[self.decoder_or_support][self.head_to_train]
            fg_cls = [fg_cls] if not isinstance(fg_cls, list) else fg_cls
            task_foreground_classes += fg_cls
            self.prune_repeat_times[self.head_to_train] = default_pruning_repeat_times

        self.task_foreground_classes = list(set(task_foreground_classes))
        self._get_device_capability()

    # ###################################################### Abstract Methods ######################################################
    @abstractmethod
    def initialize(self, predetermined_network=None, training=True, force_load_plans=False):
        pass

    @abstractmethod
    def set_oversample_rate(self):
        pass

    @abstractmethod
    def save_checkpoint(self, fname, save_optimizer=True):
        pass

    # ###################################################### Plan Processing/Loading ######################################################
    def load_plans_file(self):
        # Modified based on nnUNet v1
        def convert_keys_to_int(d: dict):
            new_dict = {}
            for k, v in d.items():
                try:
                    new_key = int(k)
                except ValueError:
                    new_key = k
                if type(v) == dict:
                    v = convert_keys_to_int(v)
                new_dict[new_key] = v
            return new_dict

        current_format = self.plans_file.split('.')[-1]
        if current_format == 'pkl':
            self.plans = load_pickle(self.plans_file)
        elif current_format == 'json':
            with open(self.plans_file, 'r') as f:
                self.plans = convert_keys_to_int(json.load(f))
        else:
            raise RuntimeError('Loading plan file format {} not supported!'.format(current_format))

    def get_plan_properties(self):
        self.intensity_properties = default_ct_intensity_properties
        if "intensityproperties" in self.plans['dataset_properties'] and self.plans['dataset_properties']['intensityproperties'] is not None:
            self.intensity_properties = self.plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = self.plans['normalization_schemes']
        self.patch_size = self.plans['plans_per_stage'][self.stage]["patch_size"]
        self.conv_per_stage = self.plans['plans_per_stage'][self.stage]["conv_kernel_sizes"]

    def process_plans(self, plans):
        # Modified based on nnUNet v1
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        # load from plans
        self.plans = plans
        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']
        for pool_op_kernel in self.net_num_pool_op_kernel_sizes:
            if self.net_pool_per_axis is None:
                self.net_pool_per_axis = [0] * len(self.net_num_pool_op_kernel_sizes[0])
            for dim, p in enumerate(pool_op_kernel):
                if p == 2:
                    self.net_pool_per_axis[dim] += 1

        self.get_plan_properties()

        self.num_input_channels = plans['num_modalities']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DON'T USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of cl-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        else:
            self.transpose_forward = plans['transpose_forward']
            self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))


    # ###################################################### Continue Training ######################################################
    def _try_to_find_model(self, is_training=False):
        # We only search for the model in the training folder.
        pth_pretrained_model = None
        if is_training and self.train_dict["model_training_setup"][self.decoder_or_support] is not None and \
                self.head_to_train in self.train_dict["model_training_setup"][self.decoder_or_support]:
            model_names = [self.task + "_" + "model_latest.model", self.task + "_" + "model_best.model", self.task + "_" + "model_final_checkpoint.model",
                           "model_latest.model", "model_best.model", "model_final_checkpoint.model"]
        else:
            model_names = [self.task + "_" + "model_final_checkpoint.model", self.task + "_" + "model_best.model", self.task + "_" + "model_latest.model",
                           "model_final_checkpoint.model", "model_best.model", "model_latest.model"]

        for model_name in model_names:
            if self.fold == "all":
                tmp_pth_pretrained_model = os.path.join(self.train_dict["output_folder_name"], self.fold, model_name)
            else:
                tmp_pth_pretrained_model = os.path.join(self.train_dict["output_folder_name"], "fold_" + str(self.fold), model_name)
            if os.path.exists(tmp_pth_pretrained_model):
                pth_pretrained_model = tmp_pth_pretrained_model
                break
        return pth_pretrained_model


    # ###################################################### Pre-trained Weight Loading ######################################################
    def load_continue_training(self, saved_model):
        self.epoch = saved_model["epoch"]
        optimizer_state_dict = saved_model["optimizer_state_dict"]
        if optimizer_state_dict is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        if saved_model["lr_scheduler_state_dict"] is not None and hasattr(self.lr_scheduler, 'state_dict'):
            self.lr_scheduler.load_state_dict(saved_model["lr_scheduler_state_dict"])

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = saved_model["plot_stuff"]
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("\033[33mWarning: Loading checkpoint: self.epoch != len(self.all_tr_losses).\033[0m")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            if self.head_to_train == "all":
                for head in self.task_dict["decoders"]:
                    self.all_val_eval_metrics[head] = self.all_val_eval_metrics[head][:self.epoch]
            else:
                if self.head_to_train in self.all_val_eval_metrics:
                    self.all_val_eval_metrics[self.head_to_train] = self.all_val_eval_metrics[self.head_to_train][:self.epoch]

    def _load_pretrained_params_ensemble(self, load_from_ema=False, is_training=True):
        # param: load_from_ema
        # Set to False during model training -- making sure that the decoder's weight is NOT loaded from the ema module
        # Set to True during testing -- making sure that the decoder's weight IS loaded from the ema module

        assert self.network is not None, "self.initialize_network must be called first"
        # set all network to CPU
        if isinstance(self.network, DDP):
            self.network = self.network.module
        self.network.to(torch.device("cpu"))
        # try to get the pretrained model path
        pth_pretrained_model = self._try_to_find_model(is_training)

        if self.task_pretrain_model_name is not None:
            # First, try to locate the model using pre-defined pth.
            if os.path.exists(self.train_dict["pretrain_model_name"]):
                pth_pretrained_model = self.train_dict["pretrain_model_name"]
            else:
                # Second, if we cannot locate the pre-defined model, then we try to locate it in the training folder.
                if self.task_dict[self.task]["fold"] == "all":
                    pth_pretrained_model = os.path.join(self.train_dict["output_folder_name"],
                                                        self.train_dict["fold"], self.task_pretrain_model_name)
                else:
                    pth_pretrained_model = os.path.join(self.train_dict["output_folder_name"],
                                                        "fold_" + str(self.train_dict["fold"]),
                                                        self.task_pretrain_model_name)

        # If the pretrained model existed -- load, else -- reset the "model_training_setup" to {"all": 1e-2, 1000}
        if pth_pretrained_model is not None and os.path.isfile(pth_pretrained_model):
            try:
                saved_model = torch.load(pth_pretrained_model, map_location=torch.device("cpu"), weights_only=False)
            except:
                saved_model = torch.load(pth_pretrained_model, map_location=torch.device("cpu"))

            self.train_dict["pretrain_model_name"] = pth_pretrained_model
            pretrained_dict = saved_model["state_dict"]
            encoder_dict = {}
            decoder_init = {}
            decoder_all_dict = {}
            supporting_all_dict = {}
            pretrained_cleaned_dict = {}
            ema_all_dict = {}
            for key, value in pretrained_dict.items():
                # remove module. prefix from DDP models
                if key.startswith("module."):
                    key = key[7:]
                    pretrained_cleaned_dict[key] = value
                # get the encoder params, if any
                if key.startswith("encoder."):
                    key = key[len("encoder."):]
                    encoder_dict[key] = value
                elif key.startswith("decoder_dict.other."):
                    key = key[len("decoder_dict.other."):]
                    decoder_init[key] = value
                # get the decoding params, if any
                elif key.startswith("decoder_dict."):
                    key = key[len("decoder_dict."):]
                    key_decoder = key.split(".")[0]
                    if key_decoder not in decoder_all_dict:
                        decoder_all_dict[key_decoder] = {}
                    key_weight = key[len(key_decoder) + 1:]
                    decoder_all_dict[key_decoder][key_weight] = value
                # get the supporting params, if any
                elif key.startswith("supporting_dict."):
                    key = key[len("supporting_dict."):]
                    key_decoder = key.split(".")[0]
                    if key_decoder not in supporting_all_dict:
                        supporting_all_dict[key_decoder] = {}
                    key_weight = key[len(key_decoder) + 1:]
                    supporting_all_dict[key_decoder][key_weight] = value
                # get the ema params, if any
                elif key.startswith("ema_dict."):
                    key = key[len("ema_dict."):]
                    key_ema = key.split(".")[0]
                    ema_flag = False
                    if "encoder_architecture_setup" in self.train_dict and self.train_dict["encoder_architecture_setup"]["enable_ema"] and \
                            key_ema == "general_encoder":
                        ema_flag = True
                    if self.train_dict["model_training_setup"]["decoders"] is not None and \
                            key_ema in self.train_dict["model_training_setup"]["decoders"] and \
                            self.train_dict["model_training_setup"]["decoders"][key_ema][-1]:
                        ema_flag = True
                    if self.train_dict["model_training_setup"]["supporting"] is not None and \
                            key_ema in self.train_dict["model_training_setup"]["supporting"] and \
                            self.train_dict["model_training_setup"]["supporting"][key_ema][-1]:
                        ema_flag = True
                    if ema_flag:
                        if key_ema not in ema_all_dict:
                            ema_all_dict[key_ema] = {}
                        key_weight = key[len(key_ema) + 1:]
                        ema_all_dict[key_ema][key_weight] = value

            # 1st, general encoder -- only load / train General Encoder once.
            if (self.train_dict["type"] == "GeneralEncoder") and (self.pretrained_network_continual_learning is None) or (self.train_dict["finetune_encoder"]):
                if self.train_dict["finetune_encoder"]:
                    self.print_to_log_file("Loading '%s' Fine-tuned GeneralEncoder -- Task '%s'" % (self.train_dict["pretrain_model_name"], self.task))
                # 1. select params for encoder
                self._load_pretrained_param_encoder(encoder_dict, pretrained_cleaned_dict)
                # 2. try to load params from if not "flag" "load_only_encoder" is False.
                # self._load_pretrained_params_encoder_ema(ema_all_dict, load_from_ema)
                # # 3. try to init each decoder using "decoder_init"
                # self._load_pretrained_params_init_decoder(decoder_init)

                if "load_only_encoder" not in self.train_dict or not self.train_dict["load_only_encoder"]:
                    self._load_pretrained_params_decoder(decoder_all_dict)
                    self._load_pretrained_params_decoder_ema(ema_all_dict, load_from_ema)
            else:
                # 2nd, decoding heads -- only loading the decoders weights
                if (self.decoder_or_support == "decoders") and (self.initial_lr is None or self.max_num_epochs is None or self.load_pretrained_decoder):
                    self._load_pretrained_params_decoder(decoder_all_dict)
                    self._load_pretrained_params_decoder_ema(ema_all_dict, load_from_ema)

                # 3rd, supporting heads -- only loading the supporting weights
                if (self.decoder_or_support == "supporting") and (self.initial_lr is None or self.max_num_epochs is None or self.load_pretrained_decoder):
                    self._load_pretrained_params_supporting(supporting_all_dict)
                    # self._load_pretrained_params_decoder_ema(ema_all_dict, load_from_ema)

                # 5th, loading all pre-trained parameters
                if self.decoder_or_support == "load_all":
                    # 1. load decoder heads
                    self._load_pretrained_params_decoder(decoder_all_dict)
                    # 2. load supporting heads
                    self._load_pretrained_params_supporting(supporting_all_dict)
                    # 3. load ema
                    self._load_pretrained_params_decoder_ema(ema_all_dict, load_from_ema)

            # only when "continue_training" is True and "same_head" is True, we try to load continue training model
            if self.train_dict["continue_training"] and self.continual_training_states["same_head"]:
                self.load_continue_training(saved_model)
            self._load_prune_stuff(saved_model)

        elif is_training:
            if self.decoder_or_support == "decoders" and self.train_dict["model_training_setup"]["supporting"] is not None \
                    and len(self.train_dict["model_training_setup"]["supporting"]) > 0:
                self.print_to_log_file("Skip training Task %s -- '%s'." % (self.task, self.decoder_or_support))
                self.print_to_log_file("Directly train the decoding head using 'supporting'")
            else:
                # If the pre-trained model is not found, then re-train "all" decoding heads using default training setup
                self.print_to_log_file("\033[31mFile Not Found: Task %s -- %s, Pretrained model is NOT found. "
                                       "Re-train the ALL decoding path using default training setup.\033[0m" %
                                       (self.task, self.decoder_or_support))
                # Reload the setting stored in "bak" -- in case users wish to use the training setup but no model found.
                if self.train_dict["type"] == "GeneralEncoder" and self.train_dict["load_only_encoder"]:
                    for key in self.train_dict["bak"]:
                        self.train_dict[key] = self.train_dict["bak"][key]
                # Reset the LR and EPOCHS of the decoders
                sampling_decay = sorted(default_sampling_decay, reverse=True)
                default_training_setup = sampling_decay + [default_lr, default_max_epoch, default_load_from_decoder,
                                                           default_prune_decoder, default_enable_decoder_ema]
                if self.train_dict["decoders"] is not None and len(self.train_dict["decoders"]) > 0 and self.decoder_or_support == "decoders":
                    self.train_dict["model_training_setup"]["decoders"] = {"all": copy.deepcopy(default_training_setup)}
                    self.oversample_foreground_percent = sampling_decay
                    self.ofp = sampling_decay[0]
                    self.initial_lr = default_lr
                    self.max_num_epochs = default_max_epoch
                    self.load_pretrained_decoder = False
                # Reset the LR and EPOCHS of the supporting
                if self.train_dict["model_training_setup"]["supporting"] is not None and len(self.train_dict["model_training_setup"]["supporting"]) > 0 \
                        and self.decoder_or_support == "supporting":
                    self.train_dict["model_training_setup"]["supporting"] = {"all": copy.deepcopy(default_training_setup)}
                    self.oversample_foreground_percent = sampling_decay
                    self.ofp = sampling_decay[0]
                    self.initial_lr = default_lr
                    self.max_num_epochs = default_max_epoch
                    self.load_pretrained_decoder = False
                del default_training_setup
                # Set "was_initialized" to False, otherwise the "trainer" will not be initialized.
                self.was_initialized = False
                # Re-initialize the "trainer"
                self.initialize()

    def _load_prune_stuff(self, saved_model):
        if "prune_stuff" in saved_model:
            if self.train_dict["continue_training"]:
                if "prune_if_is_done" in saved_model["prune_stuff"]:
                    for decoder in saved_model["prune_stuff"]["prune_if_is_done"]:
                        if decoder in self.train_dict:
                            self.prune_if_is_done[decoder] = saved_model["prune_stuff"]["prune_if_is_done"][decoder]

                if "prune_if_to_perform" in saved_model["prune_stuff"]:
                    for decoder in saved_model["prune_stuff"]["prune_if_to_perform"]:
                        if decoder in self.train_dict:
                            self.prune_if_to_perform[decoder] = saved_model["prune_stuff"]["prune_if_to_perform"][decoder]

                if "prune_ratio_in_percentage" in saved_model["prune_stuff"]:
                    self.prune_ratio_in_percentage = saved_model["prune_stuff"]["prune_ratio_in_percentage"]
                    for decoder in self.prune_ratio_in_percentage:
                        if decoder in self.train_dict[self.decoder_or_support]:
                            self.prune_ratios_to_try[decoder] = self.get_prune_ratio(default_pruning_percentages_to_try)

                if "prune_start_epoch" in saved_model["prune_stuff"]:
                    self.prune_start_epoch = saved_model["prune_stuff"]["prune_start_epoch"]

                if "prune_all_val_eval_metrics_moving_average" in saved_model["prune_stuff"]:
                    self.prune_all_val_eval_metrics_moving_average = \
                        saved_model["prune_stuff"]["prune_all_val_eval_metrics_moving_average"]

            if "prune_ratios_final_for_each_decoder" in saved_model["prune_stuff"]:
                self.prune_ratios_final_for_each_decoder = saved_model["prune_stuff"]["prune_ratios_final_for_each_decoder"]

        if self.prune_start_epoch is not None and self.prune_start_epoch <= self.epoch < self.prune_start_epoch + len(default_pruning_percentages_to_try):
            self.run_epoch_before_prune = True

    def _load_state_dict_from_ema_encoder(self, state_dict):
        if isinstance(self.network, DDP):
            network = self.network.module
        else:
            network = self.network

        try:
            network.encoder.load_state_dict(state_dict)
        except RuntimeError:
            for name, param in network.encoder.named_parameters():
                if "_orig" in name:
                    try_ema_name = name.replace("_orig", "")
                    if try_ema_name in state_dict:
                        param.data.copy_(state_dict[try_ema_name])
                else:
                    if name in state_dict:
                        param.data.copy_(state_dict[name])

    def _load_pretrained_param_encoder(self, encoder_dict, pretrained_cleaned_dict):
        init_encoder_dict = self.network.encoder.state_dict()
        if len(encoder_dict) != 0:
            encoder_dict_to_load = {k: v for k, v in encoder_dict.items()
                                    if (k in init_encoder_dict) and
                                    (init_encoder_dict[k].shape == encoder_dict[k].shape)}
        else:
            encoder_dict_to_load = {k: v for k, v in pretrained_cleaned_dict.items()
                                    if (k in init_encoder_dict) and
                                    (init_encoder_dict[k].shape == pretrained_cleaned_dict[k].shape)}

        self.print_to_log_file("Loading '%s' GeneralEncoder -- Task '%s'" % (self.train_dict["pretrain_model_name"], self.task))
        try:
            self.network.encoder.load_state_dict(encoder_dict_to_load)  # strict=True
            self.print_to_log_file("Loading General Encoder Parameters: %d / %d" % (len(init_encoder_dict), len(encoder_dict_to_load)))
        except RuntimeError:
            error_txt = "\033[31mError: Can NOT load Genernal Encoder pretrained model. Please check 1) Task name, " \
                        "2) Network Architecture, and 3) Numer of Convs Per Stage\033[0m"
            # We raise error here, because we must have the General Encoder
            raise RuntimeError(error_txt)

    def _load_pretrained_params_init_decoder(self, pre_trained_init_decoder_dict):
        if len(pre_trained_init_decoder_dict) > 0:
            # get the initial state_dict of the "self.network"
            init_decoder_dict = self.network.decoder_dict.state_dict()
            # try to load pre-trained weight for each decoder
            for task in self.task_dict:
                for current_decoder in self.task_dict[task]["decoders"]:
                    decoder_dict_to_load = {}
                    for k, v in pre_trained_init_decoder_dict.items():
                        # Try to locate the weights with matching names
                        if current_decoder + "." + k in init_decoder_dict and \
                                (init_decoder_dict[current_decoder + "." + k].shape == pre_trained_init_decoder_dict[k].shape):
                            decoder_dict_to_load[k] = v
                        # If the pre-trained model is not initialized with "pruning capability" -> missing "weight_orig"
                        # Then, we try to locate the weights with matching name that ends with "_orig"
                        if current_decoder + "." + k + "_orig" in init_decoder_dict and current_decoder + "." + k + "_orig" not in decoder_dict_to_load and \
                                (init_decoder_dict[current_decoder + "." + k + "_orig"].shape == pre_trained_init_decoder_dict[k].shape):
                            decoder_dict_to_load[k + "_orig"] = v

                    self.print_to_log_file("Initializing Decoding head -- Task '%s' - '%s'" % (self.task, current_decoder))
                    try:
                        self.network.decoder_dict[current_decoder].load_state_dict(decoder_dict_to_load, strict=False)
                    except RuntimeError:
                        error_txt = "\033[91mWarning: Decoding heads {}: Can NOT be initialized using pretrained weights, " \
                                    "using 'He-Init' instead\033[0m".format(current_decoder)
                        # We will skip loading
                        self.print_to_log_file(error_txt)

    def _load_pretrained_params_decoder(self, decoder_all_dict):
        # get the initial state_dict of the "self.network"
        init_decoder_dict = self.network.decoder_dict.state_dict()
        # try to load pre-trained weight for each decoder
        for current_decoder in self.train_dict["decoders"]:
            if current_decoder not in decoder_all_dict:
                warn_txt = "\033[31mModel Not Found: Decoding heads Task '{}'-'{}': " \
                           "Can NOT be found in the pretrained model. Please check " \
                           "1) Decoding head's name (Case Sensitive), " \
                           "2) Network Architecture, and 3) Numer of Convs Per Stage\033[0m.\n" \
                           "Yet, the decoder is not used to support other decoders. " \
                           "The training process continues.".format(self.task, current_decoder)
                # To check if the missing/unmatched decoder is used to support other organ segmentation.
                # If yes, then we raise error to stop the process. Otherwise, we will skip loading.
                for try_task in self.task_dict:
                    if "supporting" in self.task_dict[try_task]:
                        for supported_decoder in self.task_dict[try_task]["supporting"]:
                            if current_decoder in self.task_dict[try_task]["supporting"][supported_decoder]:
                                raise RuntimeError("\033[91mError: Decoding heads Task '{}'-'{}': Can NOT be found in the pretrained model. "
                                                   "It is used to support Task '{}'-'{}' segmentation"
                                                   "Please check  1) Decoding head's name (Case Sensitive), "
                                                   "2) Network Architecture, and 3) Numer of Convs Per Stage.\n"
                                                   "We 'raise' ERROR to stop the process, as it could confuse the decoder's supporting logic.\033[0m".
                                                   format(self.task, current_decoder, try_task, supported_decoder))
                self.print_to_log_file(warn_txt)
                continue
            # try to locate the pre-trained weights for the current decoder
            if self.head_to_train == "all" or current_decoder == self.head_to_train:
                decoder_dict_to_load = {}

                for k, v in decoder_all_dict[current_decoder].items():
                    # Try to locate the weights with matching names
                    if current_decoder + "." + k in init_decoder_dict and \
                            (init_decoder_dict[current_decoder + "." + k].shape == decoder_all_dict[current_decoder][k].shape):
                        decoder_dict_to_load[k] = v
                    # If the pre-trained model is not initialized with "pruning capability" -> missing "weight_orig"
                    # Then, we try to locate the weights with matching name that ends with "_orig"
                    if current_decoder + "." + k + "_orig" in init_decoder_dict and current_decoder + "." + k + "_orig" not in decoder_dict_to_load and \
                            (init_decoder_dict[current_decoder + "." + k + "_orig"].shape == decoder_all_dict[current_decoder][k].shape):
                        decoder_dict_to_load[k + "_orig"] = v
                    # If the pre-trained model is not initialized using "pruning capability" -> missing "weight_mask"
                    # Then, we try to assign the initial "weight_mask" to the "dict_to_load" for model loading completeness
                    if current_decoder + "." + k + "_mask" in init_decoder_dict and current_decoder + "." + k + "_mask" not in decoder_dict_to_load and \
                            current_decoder + "." + k + "_mask" not in decoder_all_dict:
                        decoder_dict_to_load[k + "_mask"] = init_decoder_dict[current_decoder + "." + k + "_mask"]

                self.print_to_log_file("Loading '%s' Decoding head -- Task '%s'-'%s'" %
                                       (self.train_dict["pretrain_model_name"], self.task, current_decoder))
                try:
                    self.network.decoder_dict[current_decoder].load_state_dict(decoder_dict_to_load)
                except RuntimeError:
                    warn_txt = "\033[33mWarning: Decoding heads %s: Can NOT load pretrained model. Please check " \
                                "1) Decoding head's name, " \
                                "2) Network Architecture, and 3) Numer of Convs Per Stage\033[0m" % current_decoder
                    # We will skip loading
                    self.print_to_log_file(warn_txt)

    def _load_pretrained_params_supporting(self, supporting_all_dict):
        # get the initial state_dict of the "self.network"
        init_supporting_dict = self.network.supporting_dict.state_dict()
        # try to load pre-trained weight for each decoder
        for current_decoder in self.train_dict["supporting"]:
            if current_decoder not in supporting_all_dict:
                warn_txt = "\033[31mModel Not Found: Supporting heads %s: Can NOT be found in the pretrained model. Please check " \
                           "1) Supporting head's name, " \
                           "2) Network Architecture, and 3) Numer of Convs Per Stage\033[0m" % current_decoder
                self.print_to_log_file(warn_txt)
                continue
            # try to locate the pre-trained weights for the current decoder
            if self.head_to_train == "all" or current_decoder == self.head_to_train:
                supporting_dict_to_load = {}
                for k, v in supporting_all_dict[current_decoder].items():
                    # Try to locate the weights with matching names
                    if current_decoder + "." + k in init_supporting_dict and \
                            (init_supporting_dict[current_decoder + "." + k].shape == supporting_all_dict[current_decoder][k].shape):
                        supporting_dict_to_load[k] = v
                    # If the pre-trained model is not initialized using "pruning capability" -> missing "weight_orig"
                    # Then, we try to locate the weights with matching name end with "_orig"
                    if current_decoder + "." + k + "_orig" in init_supporting_dict and current_decoder + "." + k + "_orig" not in supporting_dict_to_load and \
                            (init_supporting_dict[current_decoder + "." + k + "_orig"].shape == supporting_all_dict[current_decoder][k].shape):
                        supporting_dict_to_load[k + "_orig"] = v
                    # If the pre-trained model is not initialized with "pruning capability" -> missing "weight_mask"
                    # Then, we try to assign the initial "weight_mask" to the "dict_to_load" for model loading completeness
                    if current_decoder + "." + k + "_mask" in init_supporting_dict and current_decoder + "." + k + "_mask" not in supporting_dict_to_load and \
                            current_decoder + "." + k + "_mask" not in supporting_all_dict:
                        supporting_dict_to_load[k + "_mask"] = init_supporting_dict[current_decoder + "." + k + "_mask"]
                self.print_to_log_file("Loading '%s' Supporting head -- Task '%s' - '%s'" %
                                       (self.train_dict["pretrain_model_name"], self.task, current_decoder))
                try:
                    self.network.supporting_dict[current_decoder].load_state_dict(supporting_dict_to_load)
                except RuntimeError:
                    error_txt = "\033[33mWarning: Supporting heads %s: Can NOT load pretrained model. Please check " \
                                "1) Decoding & Supporting head's name, " \
                                "2) Network Architecture, and 3) Numer of Convs Per Stage\033[0m" % current_decoder
                    # We will skip loading
                    self.print_to_log_file(error_txt)

    def _load_pretrained_params_encoder_ema(self, ema_all_dict, load_model_weight_from_ema):
        flag_load_from_ema_from_cfg_json = False
        if self.train_dict["model_training_setup"]["decoders"] is not None:
            for decoder in self.train_dict["model_training_setup"]["decoders"]:
                if self.train_dict["model_training_setup"]["decoders"][decoder] is not None and \
                        self.train_dict["model_training_setup"]["decoders"][decoder][-1]:
                    flag_load_from_ema_from_cfg_json = True
                    break

        if flag_load_from_ema_from_cfg_json is False and self.train_dict["model_training_setup"]["supporting"] is not None:
            for support in self.train_dict["model_training_setup"]["supporting"]:
                if self.train_dict["model_training_setup"]["supporting"][support] is not None and \
                        self.train_dict["model_training_setup"]["supporting"][support][-1]:
                    break

        if "encoder_architecture_setup" in self.train_dict and self.train_dict["encoder_architecture_setup"]["enable_ema"] and len(ema_all_dict) > 0:
            init_encoder_ema = self.network.ema_dict["general_encoder"].state_dict()
            ema_dict_to_load = {}
            ema_contain_zero_weights = False
            for k, v in ema_all_dict["general_encoder"].items():
                if (k in init_encoder_ema) and (init_encoder_ema[k].shape == ema_all_dict["general_encoder"][k].shape):
                    ema_dict_to_load[k] = v
                    kernel_type = k.split(".")[-2]
                    weight_type = k.split(".")[-1]
                    if "conv" in kernel_type and "weight" in weight_type and "mask" not in weight_type:
                        if torch.all(torch.round(v * 10 ** 2) / (10 ** 2) == 0):
                            ema_contain_zero_weights = True
                            break
            if load_model_weight_from_ema:
                if ema_contain_zero_weights:
                    self.print_to_log_file("\033[33mWarning: EMA Loading General Encoder: Found zero weights. "
                                           "This will result in NO PREDICTION! Stop loading General Encoder EMA weights!\033[0m")
                else:
                    try:
                        self.network.encoder.load_state_dict(ema_dict_to_load)
                        self.print_to_log_file("EMA Loading General Encoder using Parameters: %d / %d" % (len(init_encoder_ema), len(ema_dict_to_load)))
                    except RuntimeError:
                        error_txt = "\033[33mWarning: Can NOT load EMA General Encoder pretrained model. Please check 1) Task name, " \
                                    "2) Network Architecture, and 3) Numer of Convs Per Stage\033[0m"
                        raise RuntimeError(error_txt)
            else:
                try:
                    self.network.ema_dict["general_encoder"].load_state_dict(ema_dict_to_load)
                    self.print_to_log_file("EMA Loading General Encoder Parameters: %d / %d" % (len(init_encoder_ema), len(ema_dict_to_load)))
                except RuntimeError:
                    warning_txt = "\033[33mWarning: EMA General Encoder: Can NOT load pretrained model. Please check 1) Task name, " \
                                  "2) Network Architecture, and 3) Numer of Convs Per Stage\033[0m"
                    self.print_to_log_file(warning_txt)

    def _load_pretrained_params_decoder_ema(self, ema_all_dict, load_from_ema):
        for current_decoder in self.train_dict["decoders"]:
            if self.train_dict["decoder_architecture_setup"][current_decoder]["enable_ema"] and len(ema_all_dict) > 0:
                if self.head_to_train == "all" or current_decoder == self.head_to_train:
                    init_decoder_ema = self.network.ema_dict[current_decoder].state_dict()
                    ema_dict_to_load = {}
                    if current_decoder in ema_all_dict:
                        ema_contain_zero_weights = False
                        for k, v in ema_all_dict[current_decoder].items():
                            # Try to locate the weights with matching names
                            if k in init_decoder_ema and (init_decoder_ema[k].shape == ema_all_dict[current_decoder][k].shape):
                                ema_dict_to_load[k] = v
                            # If the pre-trained model is not initialized with "pruning capability" -> missing "weight_orig"
                            # We try to locate the weights with matching name end with "_orig"
                            if k + "_orig" in init_decoder_ema and k + "_orig" not in ema_dict_to_load and \
                                    (init_decoder_ema[k + "_orig"].shape == ema_all_dict[current_decoder][k].shape):
                                ema_dict_to_load[k + "_orig"] = v
                            # If the pre-trained model is not initialized with "pruning capability" -> missing "weight_mask"
                            # We try to assign the initial "weight_mask" to the "dict_to_load" for model loading completeness
                            if k + "_mask" in init_decoder_ema and k + "_mask" not in ema_dict_to_load and \
                                    k + "_mask" not in ema_all_dict:
                                ema_dict_to_load[k + "_mask"] = init_decoder_ema[k + "_mask"]
                            kernel_type = k.split(".")[-2]
                            weight_type = k.split(".")[-1]
                            if "conv" in kernel_type and "weight" in weight_type and "mask" not in weight_type:
                                if torch.all(torch.round(v * 10 ** 2) / (10 ** 2) == 0):
                                    ema_contain_zero_weights = True
                                    break
                        flag_load_from_ema_from_cfg_json = False
                        if self.train_dict["model_training_setup"]["decoders"] is not None and \
                                current_decoder in self.train_dict["model_training_setup"]["decoders"] and \
                                self.train_dict["model_training_setup"]["decoders"][current_decoder] is not None:
                            flag_load_from_ema_from_cfg_json = \
                                flag_load_from_ema_from_cfg_json or self.train_dict["model_training_setup"]["decoders"][current_decoder][-1]

                        if not flag_load_from_ema_from_cfg_json and self.train_dict["model_training_setup"]["supporting"] is not None and \
                                current_decoder in self.train_dict["model_training_setup"]["supporting"] and \
                                self.train_dict["model_training_setup"]["supporting"][current_decoder] is not None:
                            flag_load_from_ema_from_cfg_json = \
                                flag_load_from_ema_from_cfg_json or self.train_dict["model_training_setup"]["supporting"][current_decoder][-1]

                        if load_from_ema:
                            if ema_contain_zero_weights:
                                self.print_to_log_file("\033[33mWarning: EMA Loading Decoding head %s: Found zero weights. "
                                                       "This will result in NO PREDICTION! Stop loading EMA weights!\033[0m" % current_decoder)
                            elif not flag_load_from_ema_from_cfg_json:
                                self.print_to_log_file("\033[33mWarning: EMA weights are NOT loaded to Decoder %s "
                                                       "The EMA updating is not enabled in the cfg JSON file.\033[0m" % current_decoder)
                            else:
                                # try to load the weights from ema for the decoder. If it fails, then we will skip
                                try:
                                    self.network.decoder_dict[current_decoder].load_state_dict(ema_dict_to_load)
                                    self.print_to_log_file("Loading '%s' EMA to Decoder -- Task '%s' - '%s'" %
                                                           (self.train_dict["pretrain_model_name"], self.task, current_decoder))
                                except RuntimeError:
                                    error_txt = "\033[33mWarning: EMA loading heads %s: Can NOT load pretrained EMA model. Please check " \
                                                "1) Decoding head's name, " \
                                                "2) Network Architecture, and 3) Numer of Convs Per Stage\033[0m" % current_decoder
                                    # We will skip loading
                                    self.print_to_log_file(error_txt)
                        else:
                            # try to load the weight from ema dict.
                            try:
                                self.network.ema_dict[current_decoder].load_state_dict(ema_dict_to_load)
                                self.print_to_log_file("Loading '%s' EMA to EMA -- Task '%s' - '%s'" %
                                                       (self.train_dict["pretrain_model_name"], self.task, current_decoder))
                            except RuntimeError:
                                error_txt = "\033[33mWarning: EMA loading heads %s: Can NOT load pretrained EMA model. Please check " \
                                            "1) Decoding head's name, " \
                                            "2) Network Architecture, and 3) Numer of Convs Per Stage\033[0m" % current_decoder
                                # We will skip loading
                                self.print_to_log_file(error_txt)

    # ###################################################### Model Initialization ######################################################
    def initialize_network(self, is_ddp=False):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.pretrained_network_continual_learning is not None:
            if isinstance(self.pretrained_network_continual_learning, DDP):
                self.network = self.pretrained_network_continual_learning.module
            else:
                self.network = self.pretrained_network_continual_learning
        else:
            if self.threeD:
                conv_op = nn.Conv3d
                dropout_op = nn.Dropout3d
                norm_op = nn.InstanceNorm3d
            else:
                conv_op = nn.Conv2d
                dropout_op = nn.Dropout2d
                norm_op = nn.InstanceNorm2d

            norm_op_kwargs = {"eps": 1e-5, "affine": True}
            dropout_op_kwargs = {"p": 0, "inplace": True}
            net_nonlin = nn.LeakyReLU
            net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
            # we only initialize the network architecture once
            self.network = Generic_UNet_Continual_Base(
                self.task_dict, self.num_input_channels, self.base_num_features, num_pool=len(self.net_num_pool_op_kernel_sizes),
                feat_map_mul_on_downscale=2, conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs, nonlin=net_nonlin, nonlin_kwargs=net_nonlin_kwargs, deep_supervision=True,
                dropout_in_localization=False, final_nonlin=lambda x: x, weight_initializer=InitWeights_He(1e-2), upscale_logits=False,
                convolutional_pooling=True, convolutional_upsampling=True)
            # Initialize the network with pruning capability -> adding "weight_orig" and "weight_mask", and change the hook from "weight" to "weight_orig"
            perform_network_initialization_with_pruning_capability(self.network)
            self.network.inference_apply_nonlin = softmax_helper


    def initialize_prune_ratio(self):
        if isinstance(self.network, DDP):
            network = self.network.module
        else:
            network = self.network

        for decoder in network.decoder_dict:
            if decoder in self.train_dict[self.decoder_or_support]:
                if decoder not in self.prune_ratio_in_percentage:
                    self.prune_ratio_in_percentage[decoder] = self._get_prune_percentage_based_on_default_base_num_feature(decoder)
                self.prune_ratios_to_try[decoder] = self.get_prune_ratio(self.prune_ratio_in_percentage[decoder])

    # ###################################################### Data loader Initialization ######################################################
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        tr_max_num_batches = (self.max_num_epochs * self.num_train_batches_per_epoch) // self.data_aug_params.get("num_threads")
        # Need to know current epoch to correctly set current oversample foreground percent if doing continue training
        if self.continual_training_states is not None:
            current_epoch = self.continual_training_states["continue_training_epoch"]
        else:
            current_epoch = 0
        curruent_tr_batch_idx = (current_epoch * self.num_train_batches_per_epoch) // self.data_aug_params.get("num_threads")

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode="r",
                                 oversample_foreground_class=self.task_foreground_classes,
                                 head_info=(self.head_to_train, self.decoder_or_support, self.plans["num_classes"]),
                                 train_dict=self.train_dict, max_num_batches=tr_max_num_batches,
                                 curr_batch_idx=curruent_tr_batch_idx)
            # the foreground oversampling rate for the evaluation is always 0.33 (same as nnUNet)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=tuple([0.33, 0.33]), oversample_percent_decay=False,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode="r",
                                  head_info=(self.head_to_train, self.decoder_or_support, self.plans["num_classes"]),
                                  train_dict=self.train_dict)
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode="r")
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode="r")
        return dl_tr, dl_val

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def setup_data_aug_params(self):
        self.deep_supervision_scales = \
            [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0] if self.head_to_train != 'all' else None
        self.data_aug_params['patch_size_for_spatialtransform'] = np.array(self.patch_size)
        if self.train_dict["no_mirroring"]:
            self.data_aug_params["do_mirror"] = False
        else:
            self.data_aug_params["do_mirror"] = True
        self.data_aug_params["num_cached_per_thread"] = max(6, default_num_threads // 2)
        self.data_aug_params["normalization_schemes"] = self.normalization_schemes


    # ###################################################### Inference Preprocess Methods ######################################################
    def preprocess_patient(self, input_files, seg_files=None, bpr_range=None):
        from clnet.training.model_restore import recursive_find_python_class
        preprocessor_name = self.plans.get('preprocessor_name')
        if preprocessor_name is None:
            if self.threeD:
                preprocessor_name = "GenericPreprocessor"
            else:
                preprocessor_name = "PreprocessorFor2D"

        # print("using preprocessor", preprocessor_name)
        preprocessor_class = recursive_find_python_class([join(clnet.__path__[0], "preprocessing")], preprocessor_name, current_module="clnet.preprocessing")
        assert preprocessor_class is not None, "Could not find preprocessor %s in clNet.preprocessing" % preprocessor_name
        preprocessor = preprocessor_class(self.normalization_schemes, self.use_mask_for_norm, self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files, self.plans['plans_per_stage'][self.stage]['current_spacing'], seg_files, bpr_range)
        return d, s, properties

    def predict_preprocessed_data_return_seg_and_softmax_ensemble(
            self, data: np.ndarray, decoder_or_support: str, head_to_train: str, do_mirroring: bool = True,
            mirror_axes: Tuple[int] = None, use_sliding_window: bool = True, step_size: float = 0.5,
            use_gaussian: bool = True, pad_border_mode: str = 'constant', pad_kwargs: dict = None,
            all_in_gpu: bool = False, verbose: bool = True, mixed_precision=True,
            current_patch_size: list = None, is_inference=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        if do_mirroring:
            print("WARNING! do_mirroring was True but we set do_mirroring to False")
            do_mirroring = False

        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network

        ds = net.do_ds
        net.set_do_ds(False)  # ! set do_ds=False for all decoders

        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        current_mode = self.network.training
        if current_patch_size is None or (isinstance(current_patch_size, list) and len(current_patch_size) != 3):
            # In training, the "current_patch_size" is always None -> we will self.patch_size for training.
            # In inference, we will try to load current_patch_size from the cfg.json
            current_patch_size = self.patch_size
        self.network.eval()
        ram_in_byte = sum(p.numel() for p in net.parameters()) * 4 * 3
        ret = net.predict_3D_ensemble(data, self.train_dict, decoder_or_support, head_to_train, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size, patch_size=current_patch_size,
                                      regions_class_order=self.regions_class_order, use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose, mixed_precision=mixed_precision, ram_in_byte=ram_in_byte)

        if not is_inference:
            self.network.train(current_mode)
            # set to eval mode
            self.network.set_to_eval_mode(self.train_dict, self.decoder_or_support, self.head_to_train)
            self.network.set_do_ds(ds)  # ! recover do_ds
        return ret


    # ###################################################### Other Methods ######################################################
    def _plot_network_architecture(self):
        if isinstance(self.network, DDP):
            network = self.network.module
        else:
            network = self.network
        if OptimizedModule is not None:
            if isinstance(network, OptimizedModule):
                network = network._orig_mod
        # Always print out the general encoder architecture
        self.print_to_log_file(network.encoder)
        if self.head_to_train == "all":
            if self.decoder_or_support == "decoders":
                for decoder in self.train_dict["decoders"]:
                    self.print_to_log_file("%s - %s Network Architecture" % (decoder, self.decoder_or_support))
                    self.print_to_log_file(network.decoder_dict[decoder])
            else:
                for decoder in self.train_dict["supporting"]:
                    self.print_to_log_file("%s - %s Network Architecture" % (decoder, self.decoder_or_support))
                    self.print_to_log_file(network.supporting_dict[decoder])
        else:
            if self.decoder_or_support == "decoders":
                self.print_to_log_file("%s - %s Network Architecture" % (self.head_to_train, self.decoder_or_support))
                self.print_to_log_file(network.decoder_dict[self.head_to_train])
            else:
                self.print_to_log_file("%s - %s Network Architecture" % (self.head_to_train, self.decoder_or_support))
                self.print_to_log_file(network.supporting_dict[self.head_to_train])

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, join(self.output_folder, "debug.json"))
        if "encoder_architecture_setup" in self.train_dict:
            if "num_conv_per_stage" in self.train_dict["encoder_architecture_setup"]:
                conv_per_stage = self.train_dict["encoder_architecture_setup"]["num_conv_per_stage"]
                self.plans["plans_per_stage"]["conv_per_stage"] = conv_per_stage
        import shutil
        try:
            shutil.copy(self.plans_file, join(self.output_folder_base, "plans.{}".format(self.plans_file.split('.')[-1])))
        except shutil.SameFileError:
            pass

    def update_fold(self, fold):
        """
        used to swap between folds for inference (ensemble of models from cross-validation)
        DO NOT USE DURING TRAINING AS THIS WILL NOT UPDATE THE DATASET SPLIT AND THE DATA AUGMENTATION GENERATORS
        :param fold:
        :return:
        """
        if fold is not None:
            if isinstance(fold, str):
                assert fold == "all", "if self.fold is a string then it must be \'all\'"
                if self.output_folder.endswith("%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "%s" % str(fold))
            else:
                if self.output_folder.endswith("fold_%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "fold_%s" % str(fold))
            self.fold = fold

    def get_prune_ratio(self, prune_ratio_in_percentage):
        prune_ratios_to_try = []
        for i, percentage in enumerate(prune_ratio_in_percentage):
            if i == 0:
                prune_ratios_to_try.append(percentage / 100.0)
            else:
                prune_ratios_to_try.append(1 - (1 - percentage / 100) / (1 - prune_ratio_in_percentage[i - 1] / 100.))
        return prune_ratios_to_try

    def check_decoder_sparsity(self):
        if isinstance(self.network, DDP):
            network = self.network.module
        else:
            network = self.network

        for decoder in self.train_dict[self.decoder_or_support]:
            self.prune_decoder_sparsity_before_prune[decoder] = self.count_masking_ratio(network.decoder_dict[decoder])
            if self.prune_decoder_sparsity_before_prune[decoder] > default_pruning_percentages_to_try[0]:
                self.prune_ratio_in_percentage[decoder].append(self.prune_decoder_sparsity_before_prune[decoder])
                self.prune_ratio_in_percentage[decoder] = sorted(self.prune_ratio_in_percentage[decoder])
                self.prune_ratio_in_percentage[decoder].pop(0)
            elif 0 < self.prune_decoder_sparsity_before_prune[decoder] < default_pruning_percentages_to_try[0]:
                self.prune_ratio_in_percentage[decoder][0] = self.prune_decoder_sparsity_before_prune[decoder]

    def reset_pruning_mask(self, head):
        if isinstance(self.network, DDP):
            decoder = self.network.module.decoder_dict[head]
        else:
            decoder = self.network.decoder_dict[head]
        for name, param in decoder.named_buffers():
            if name.endswith('weight_mask'):
                with torch.no_grad():
                    param.fill_(1.0)
            if name.endswith('bias_mask'):
                with torch.no_grad():
                    param.bias_mask.fill_(1.0)

    def count_masking_ratio(self, model):
        if isinstance(model, DDP):
            model = model.module
        zero_count = 0
        one_count = 0
        try:
            if hasattr(model, "named_buffers"):
                for name, param in model.named_buffers():
                    if "_mask" in name:
                        zero_count += torch.sum(param.data == 0)
                        one_count += torch.sum(param.data != 0)
                if zero_count != 0 or one_count != 0:
                    masking_ratio = zero_count / (zero_count + one_count) * 100.0
                    if torch.cuda.is_available():
                        return float(masking_ratio.cpu().numpy())
                    else:
                        return float(masking_ratio.numpy())
            else:
                for name in model:
                    if "_mask" in name:
                        zero_count += torch.sum(model[name] == 0)
                        one_count += torch.sum(model[name] != 0)
                if zero_count != 0 or one_count != 0:
                    masking_ratio = zero_count / (zero_count + one_count) * 100.0
                    if torch.cuda.is_available():
                        return float(masking_ratio.cpu().numpy())
                    else:
                        return float(masking_ratio.numpy())
        except RuntimeError:
            self.print_to_log_file("\033[33mWaring: Can NOT measure masking percentage\033[0m")

    def check_model_similarity(self, model1, model2):
        mask_same = "Same"
        parm_same = "Same"
        if hasattr(model1, "named_buffers"):
            model2_dict = dict(model2.named_buffers())
            for name, param in model1.named_buffers():
                if name in model2_dict:
                    if not torch.equal(param.data, model2_dict[name]):
                        mask_same = "Diff"

        if hasattr(model1, "named_parameters"):
            model2_dict = dict(model2.named_parameters())
            for name, param in model1.named_parameters():
                if name in model2_dict:
                    if not torch.equal(param.data, model2_dict[name]):
                        parm_same = "Diff"

        self.print_to_log_file("Mask -- ", mask_same, "Param -- ", parm_same)

    def gen_target_dict_by_heads(self, target, head_to_train=None):
        ret_target = {}
        if head_to_train is None:
            head_to_train = self.head_to_train
        if head_to_train == "all":
            head_list = list(self.train_dict["decoders"].keys()) if self.decoder_or_support == "decoders" else list(self.train_dict["supporting"].keys())
            for idx, head in enumerate(head_list):
                ret_target[head] = [target[s][:, [idx], :] for s in range(len(target))]
        else:
            ret_target[head_to_train] = target

        return ret_target

    def _plot_progress(self):
        """
        Should probably be improved
        :return:
        """
        try:
            font = {'weight': 'normal', 'size': 18}
            matplotlib.rc('font', **font)
            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            ave_all_val_eval_metrics = [0] * len(x_values)
            for i in range(len(x_values)):
                for head in self.all_val_eval_metrics:
                    ave_all_val_eval_metrics[i] += self.all_val_eval_metrics[head][i]
                ave_all_val_eval_metrics[i] /= len(self.all_val_eval_metrics)
            if len(ave_all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, ave_all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig_output_name = "progress_" + self.task + "_" + self.head_to_train + "_" + \
                              self.decoder_or_support + ".png"
            fig.savefig(join(self.output_folder, fig_output_name))
            plt.close()
        except IOError:
            print("failed to plot: ", sys.exc_info())

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = ("%s:" % dt_object, *args)

            if self.log_file is None:
                maybe_mkdir_p(self.output_folder)
                timestamp = datetime.now()
                self.log_file = join(self.output_folder, "training_log_%s_%s_%s_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                     (self.task, self.head_to_train, self.decoder_or_support, timestamp.year,
                                      timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second))
                with open(self.log_file, 'w') as f:
                    f.write("Starting... \n")
            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)

    def _get_prune_percentage_based_on_default_base_num_feature(self, decoder):
        # Get the pruning percentage based on the "default_base_num_feature"
        # If the "default_base_num_feature" is less than the decoder's base_num_feature, then we keep the pruning ratio unchanged.
        # Else, we change the decoder's pruning ratio based on the decoder's base_num_feature, s.t., the pruning ratio is TOO MUCH for the decoder.
        # E.g., if decoder's base_num_feature=16, then, we change the default 80% pruning ratio to 60%.
        pruning_percentages_to_try = copy.deepcopy(default_pruning_percentages_to_try)
        if decoder not in self.train_dict["decoder_architecture_setup"]:
            return pruning_percentages_to_try
        decoder_ratio_compare_to_full_decoder = self.train_dict["decoder_architecture_setup"][decoder]["base_num_feature"]
        if 0 < decoder_ratio_compare_to_full_decoder < default_base_num_feature:
            ret_percentages = []
            for prune_ratio in pruning_percentages_to_try:
                try_percentage = 100 - default_base_num_feature * (100 - prune_ratio) / decoder_ratio_compare_to_full_decoder
                ret_percentages.append(try_percentage)
            return ret_percentages
        else:
            return pruning_percentages_to_try

    def _get_prune_num_batches_per_epoch(self, decoder, current_pruning_percentage, to_perform=True):
        # We set the batch size of the first pruning epoch is the same as the default training epoch
        # The pruning training epoch increases linearly as it progress towards higher pruning ratios.
        # The maximum number of training proportion is 3x, i.e., if 256x3 = 768
        if self.prune_if_to_perform[decoder] and to_perform and to_perform:
            o = (3 * self.prune_ratio_in_percentage[decoder][0] - self.prune_ratio_in_percentage[decoder][-1]) / 2
            p = 2 / (self.prune_ratio_in_percentage[decoder][-1] - self.prune_ratio_in_percentage[decoder][0])
            current_num_batches_per_epoch = np.round(self.backup_num_batches_per_epoch * max(1, (current_pruning_percentage - o) * p))
            current_val_num_batches_per_epoch = np.round(self.backup_num_val_batches_per_epoch * max(1, (current_pruning_percentage - o) * p))
            return int(current_num_batches_per_epoch), int(current_val_num_batches_per_epoch)
        else:
            return self.num_train_batches_per_epoch, self.num_val_batches_per_epoch

    def _get_eval_moving_average_before_pruning(self, window_size=default_pruning_percentile_moving_average_window_size):
        for head in self.all_val_eval_metrics:
            dsc_percentile_before_pruning = self.all_val_eval_metrics[head][:self.prune_start_epoch]
            # We consider the moving average of the last 2 * window_size epochs. If the length is less than 2 * window_size, then we use the last epoch.
            if len(dsc_percentile_before_pruning) >= 2 * window_size:
                moving_ave = np.convolve(dsc_percentile_before_pruning, np.ones(window_size) / window_size, mode='valid')
            else:
                if len(dsc_percentile_before_pruning) == 0:
                    moving_ave = [0]
                else:
                    moving_ave = [dsc_percentile_before_pruning[-1]]
            self.prune_all_val_eval_metrics_moving_average[head] = moving_ave[-1]

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None:
            if torch.cuda.is_available():
                self.amp_grad_scaler = torch.cuda.amp.GradScaler()
            else:
                self.amp_grad_scaler = None

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal', 'size': 18}
            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")
            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)
            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()

        except IOError:
            self.print_to_log_file("Failed to plot: ", sys.exc_info())

    def _get_device_capability(self):
        compute_capability = np.inf
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                compute_capability = min(torch.cuda.get_device_capability(i)[0], compute_capability)
        else:
            compute_capability = None
        self.compute_capability = compute_capability

    def _reset_momentum(self):
        # this is resolve the NaN issue when switching from "warm up" to "poly" lr scheduler.
        if isinstance(self.optimizer, torch.optim.SGD):
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if 'momentum_buffer' in self.optimizer.state[param]:
                        del self.optimizer.state[param]['momentum_buffer']
        else:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    opt_state = self.optimizer.state[param]
                    if 'exp_avg' in opt_state:
                        opt_state['exp_avg'].zero_()
                    if 'exp_avg_sq' in opt_state:
                        opt_state['exp_avg_sq'].zero_()

    def convert_to_sparse(self):
        if self.epoch == self.prune_start_epoch + len(default_pruning_percentages_to_try):
            # First, we need to restore the compiled network to its original mode
            if torch.cuda.is_available():
                if OptimizedModule is not None:
                    if isinstance(self.network, OptimizedModule):
                        self.network = self.network._orig_mod
            # Then, we check if the network is wrapped using DDP.
            if isinstance(self.network, DDP):
                network = self.network.module
            else:
                network = self.network
            device_id = next(network.parameters()).device
            # Converting the pruned network to its corresponding sparse version
            if self.head_to_train == "all":
                for head in self.train_dict[self.decoder_or_support]:
                    if self.prune_if_is_done[head] and self.flag_convert_to_sparse[head]:
                        network.decoder_dict[head] = ModelSparse(network.decoder_dict[head]).to(device_id)
                        network.ema_dict[head] = ModelSparse(network.ema_dict[head]).to(device_id)
                        self.flag_convert_to_sparse[head] = False
            else:
                if self.prune_if_is_done[self.head_to_train] and self.flag_convert_to_sparse[self.head_to_train]:
                    network.decoder_dict[self.head_to_train] = ModelSparse(network.decoder_dict[self.head_to_train]).to(device_id)
                    network.decoder_dict[self.head_to_train] = ModelSparse(network.ema_dict[self.head_to_train]).to(device_id)
                    self.flag_convert_to_sparse[self.head_to_train] = False
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.initialize_optimizer_and_scheduler(current_lr)

    def compile_network(self, compile_loss=True):
        if torch.cuda.is_available():
            if OptimizedModule is not None:
                if isinstance(self.network, OptimizedModule):
                    self.network = self.network._orig_mod
                if isinstance(self.loss, OptimizedModule):
                    self.loss = self.loss._orig_mod
                try:
                    if self.compute_capability <= 6:
                        self.network = torch.compile(self.network, backend='aot_eager')
                        if compile_loss:
                            self.loss = torch.compile(self.loss, backend='aot_eager')
                    else:
                        try:
                            self.network = torch.compile(self.network)
                            if compile_loss:
                                self.loss = torch.compile(self.loss)
                        except RuntimeError:
                            self.network = torch.compile(self.network, backend='eager')
                            if compile_loss:
                                self.loss = torch.compile(self.loss, backend='eager')
                    self.print_to_log_file("Network compiled. CUDA Capability", self.compute_capability)
                except RuntimeError:
                    self.print_to_log_file("The CL-Net is NOT compiled! Please check PyTorch and CUDA might not be compatible.")
            else:
                self.print_to_log_file("The CL-Net is NOT compiled! Module '_dynamo' is not Loaded!")
        else:
            self.print_to_log_file("The CL-Net is NOT compiled! Please check CUDA availability!")
