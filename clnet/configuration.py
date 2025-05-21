import multiprocessing

# ############### System setup ###############
num_total_threads = multiprocessing.cpu_count()
default_num_threads = min(12, num_total_threads - 2)

# ############### Trainer default setup ###############
default_trainer = "clNetTrainerV2_SelectiveChannelDA_Continual_Decoding_Ensemble"
default_network = "3d_fullres"
default_optimizer = "adam"

# ############### CNN default setup ###############
default_num_pool_per_axis = [4, 5, 5]
default_pool = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
default_conv = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
default_num_conv_per_stage = [2, 2, 2, 2, 2, 2]
default_base_num_feature = 32
default_max_num_features = 320
default_ge_basic_block = "conv_block"
default_decoder_basic_block = "conv_block"

# ############### CT intensity default setup ###############
default_ct_intensity_properties = {
    0: {"median": 29, "mean": -187.85325622558594, "sd": 490.7054443359375, "mn": -2048, "mx": 1024,
        "percentile_99_5": 1265.0, "percentile_00_5": -995.0}
}
# ############### Model training default setup ###############
default_load_from_decoder = True
default_decoder_update = True
default_warmup_epoch = 5
default_patch_size = [112, 96, 128]
default_batch_size = 2
default_weight_decay = 3e-5
default_num_train_batches_per_epoch = 256
default_num_val_batches_per_epoch = 48
# default_num_train_batches_per_epoch = 5
# default_num_val_batches_per_epoch = 5
default_save_every = 25
# default_save_every = 1
default_lr = 3e-4
default_max_epoch = 1000
default_sampling_decay = [0.3, 0.35]
default_enable_encoder_ema = False
default_enable_decoder_ema = True
default_alpha_ema_encoder = 0.999
default_alpha_ema_decoder = 0.99
default_gpu_ram_constraint = 16

# ############### Pruning default setup ###############
default_prune_decoder = True
default_pruning_extending_num_batches_per_epoch = False
default_pruning_percentages_to_try = [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 99, 99.2, 99.4, 99.6, 99.8]
# default_pruning_percentages_to_try = [80, 92]
default_pruning_performance_drop_threshold = 0.02
default_pruning_performance_recovery_epochs = 50
# default_pruning_performance_recovery_epochs = 5
default_pruning_percentile_moving_average_window_size = 10
default_pruning_repeat_times = 10

# ############### determines what threshold to use for resampling the low resolution axis separately (with NN)
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3
