import os.path
import clnet
import shutil
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import *

from clnet.paths import *
from clnet.configuration import *
from clnet.bpreg.bpr_pred import bpr_gen
from clnet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from clnet.experiment_planning.utils import crop, collect_bpr_scores_per_label
from clnet.utilities.task_name_id_conversion import convert_id_to_task_name
from clnet.preprocessing.sanity_checks import verify_dataset_integrity
from clnet.training.model_restore import recursive_find_python_class


def main():
    import argparse
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="List of integers belonging to the task ids you wish to run"
                                                            " experiment planning and preprocessing for. Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder")
    parser.add_argument("-pl3d", "--planner3d", type=str, default="ExperimentPlanner3D",
                        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
                             "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
                             "configured")
    parser.add_argument("-pl2d", "--planner2d", type=str, default="ExperimentPlanner2D",
                        help="Name of the ExperimentPlanner class for the 2D U-Net. Default is ExperimentPlanner2D_v21. "
                             "Can be 'None', in which case this U-Net will not be configured")
    parser.add_argument("-overwrite_cropping", required=False, default=False, action="store_true",
                        help="Set this flag if you dont want to overwrite cropping")
    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("-overwrite_plans", type=str, default=None, required=False,
                        help="Use this to specify a plans file that should be used instead of whatever nnU-Net would "
                             "configure automatically. This will overwrite everything: intensity normalization, "
                             "network architecture, target spacing etc. Using this is useful for using pretrained "
                             "model weights as this will guarantee that the network architecture on the target "
                             "dataset is the same as on the source dataset and the weights can therefore be transferred.\n"
                             "Pro tip: If you want to pretrain on Hepaticvessel and apply the result to LiTS then use "
                             "the LiTS plans to run the preprocessing of the HepaticVessel task.\n"
                             "Make sure to only use plans files that were "
                             "generated with the same number of modalities as the target dataset (LiTS -> BCV or "
                             "LiTS -> Task008_HepaticVessel is OK. BraTS -> LiTS is not (BraTS has 4 input modalities, "
                             "LiTS has just one)). Also only do things that make sense. This functionality is beta with"
                             "no support given.\n"
                             "Note that this will first print the old plans (which are going to be overwritten) and "
                             "then the new ones (provided that -no_pp was NOT set).")
    parser.add_argument("-overwrite_plans_identifier", type=str, default=None, required=False,
                        help="If you set overwrite_plans you need to provide a unique identifier so that nnUNet knows "
                             "where to look for the correct plans and data. Assume your identifier is called "
                             "IDENTIFIER, the correct training command would be:\n"
                             "'nnUNet_train CONFIG TRAINER TASKID FOLD -p nnUNetPlans_pretrained_IDENTIFIER "
                             "-pretrained_weights FILENAME'")

    args = parser.parse_args()
    task_ids = args.task_ids
    overwrite_cropping = args.overwrite_cropping
    dont_run_preprocessing = args.no_pp
    planner_name3d = args.planner3d
    planner_name2d = args.planner2d

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    if args.overwrite_plans is not None:
        if planner_name2d is not None:
            print("Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
                  "skip 2d planning and preprocessing.")
        assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', "When using --overwrite_plans you need to use " \
                                                                       "'-pl3d ExperimentPlanner3D_v21_Pretrained'"

    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        task_name = convert_id_to_task_name(i)

        if args.verify_dataset_integrity:
            verify_dataset_integrity(join(clNet_raw_data, task_name))
        print("Preprocessing -- Cropping")
        crop(task_name, overwrite_cropping, default_num_threads)

        ct_files = subfiles(join(clNet_raw_data, task_name, 'imagesTr'), join=True, suffix='_0000.nii.gz')
        bpr_files = [join(clNet_cropped_data, task_name, os.path.basename(f).replace('.nii.gz', '.json')) for f in ct_files]
        bpr_gen(ct_files, bpr_files, overwrite_existing=overwrite_cropping)
        dataset_json = load_json(join(clNet_cropped_data, task_name, 'dataset.json'))
        classes = dataset_json['labels']
        print("Preprocessing -- Analyzing Body Part Scores")
        collect_bpr_scores_per_label(task_name, classes, overwrite=True)

        tasks.append(task_name)

    search_in = join(clnet.__path__[0], "experiment_planning")

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="clnet.experiment_planning")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in clnet.experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(clNet_cropped_data, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)

        dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
        modalities = list(dataset_json["modality"].values())

        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False

        # this class creates the fingerprint
        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=default_num_threads)
        # this will write output files that will be used by the ExperimentPlanner
        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)

        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "bpr_scores.json"), preprocessing_output_dir_this_task)
        shutil.copy(join(clNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (default_num_threads, default_num_threads)

        print("number of threads: ", threads, "\n")

        if planner_3d is not None:
            if args.overwrite_plans is not None:
                assert args.overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
                exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, args.overwrite_plans, args.overwrite_plans_identifier)
            else:
                exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:
                exp_planner.run_preprocessing(threads)

        print("Done: Data preprocessing for task", t)


if __name__ == "__main__":
    main()
