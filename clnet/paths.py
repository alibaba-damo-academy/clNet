import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# do not modify these unless you know what you are doing
my_output_identifier = "clNet"
default_plans_identifier = "clNetPlans"
default_data_identifier = 'clNetData_plans'
default_trainer = "clNetTrainerV2_SelectiveChannelDA"


base = "/mnt/nas/suyanzhou.syz/clnet_project/pretrained_model"
# base = os.environ['clNet_raw_data_base'] if "clNet_raw_data_base" in os.environ.keys() else None
preprocessing_output_dir = os.path.join(base, "preprocessed")
network_training_output_dir_base = os.path.join(base, "results")
default_bpreg_model = os.path.join(network_training_output_dir_base, "public_bpr_model")

if base is not None:
    clNet_raw_data = join(base, "clNet_raw_data")
    clNet_cropped_data = join(base, "clNet_cropped_data")
    maybe_mkdir_p(clNet_raw_data)
    maybe_mkdir_p(clNet_cropped_data)
else:
    print("clNet_raw_data_base is not defined.")
    clNet_cropped_data = clNet_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("clNet_preprocessed is not defined.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined.")
    network_training_output_dir = None
