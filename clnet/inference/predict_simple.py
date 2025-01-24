#   Author @Dazhou Guo + @Puyang Wang
#   Data: 06.16.2023


import argparse
from clnet.configuration import *
from clnet.inference.predict_simple_deliver import clnet_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('continual_decoding_ensemble_setup',
                        help='path of the json file.'
                             'containing previously trained models for decoding ensemble trainer')

    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)

    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")

    parser.add_argument("--num_threads_preprocessing", required=False, default=default_num_threads, type=int,
                        help="Determines many background processes will be used for data preprocessing. "
                             "Reduce this if you run into out of memory (RAM) problems. Default: 6")

    parser.add_argument("--num_threads_nifti_save", required=False, default=default_num_threads, type=int,
                        help="Determines many background processes will be used for segmentation export. "
                             "Reduce this if you run into out of memory (RAM) problems. Default: 2")

    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")

    parser.add_argument("--expected_num_modalities", required=False, default=1, type=int,
                        help="Determines how many modalities are expected in the input folder. "
                             "If this is not the case, the script will exit with an error. Default: 1")

    parser.add_argument("--overwrite_existing_preprocessing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")

    parser.add_argument("--overwrite_existing_pred", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")

    parser.add_argument("--compile_network", type=str, default=False, required=False, action="store_true",
                        help="if or not to compile the network for potential faster inference")

    parser.add_argument("--all_in_gpu", type=str, default="True", required=False, help="can be None, False or True")
    parser.add_argument("--fast", required=False, default=False, action="store_true", help="Enable to use fast inference mode. "
                                                                                           "It may degrade the prediction accuracy.")
    parser.add_argument("--save_intermediate_result_for_debug", required=False, default=False, action="store_true",
                        help="Set this flag if to save the intermediate resampled images for debug.")

    parser.add_argument("--task_plan_to_use", type=str, default="", required=False,
                        help="Select specific 'plans.json' in a designated folder for trainer initialization."
                             "The default is GeneralEncoder's 'plans.json' file. ")

    parser.add_argument("--step_size", type=float, default=0.8, required=False, help="don't touch")

    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')

    parser.add_argument("--do_not_inference", required=False, default=False, action="store_true",
                        help="Set this flag if you do not wish to perform inference")

    parser.add_argument("--disable_bm_bpr", required=False, default=False, action="store_true",
                        help="Set this flag if you do not wish to perform coarse body masking (BM) and body part regression (BPR).")

    args = parser.parse_args()
    task_plan_to_use = args.task_plan_to_use
    clnet_cfg = args.continual_decoding_ensemble_setup
    input_folder = args.input_folder
    output_folder = args.output_folder
    all_in_gpu = args.all_in_gpu
    fast = args.fast

    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = args.disable_tta
    expected_num_modalities = args.expected_num_modalities
    step_size = args.step_size
    overwrite_existing_preprocessing = args.overwrite_existing_preprocessing
    overwrite_existing_pred = args.overwrite_existing_pred
    save_intermediate_result_for_debug = args.save_intermediate_result_for_debug
    compile_network = args.compile_network
    disable_mixed_precision = args.disable_mixed_precision
    do_not_inference = args.do_not_inference
    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = False
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    clnet_inference(clnet_cfg, input_folder, output_folder, task_plan_to_use, num_threads_preprocessing,
                    disable_tta, expected_num_modalities, num_threads_nifti_save, overwrite_existing_preprocessing,
                    fast, overwrite_existing_pred, compile_network, save_intermediate_result_for_debug,
                    all_in_gpu, step_size, disable_mixed_precision, do_not_inference)


if __name__ == "__main__":
    main()
