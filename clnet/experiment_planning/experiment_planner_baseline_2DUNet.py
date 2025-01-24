#   Author @Dazhou Guo
#   Data: 07.12.2023

from copy import deepcopy
from clnet.experiment_planning.common_utils import get_pool_and_conv_props
from clnet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner3D
from clnet.network_architecture.generic_UNet import Generic_UNet
from clnet.paths import *
import numpy as np


class ExperimentPlanner2D(ExperimentPlanner3D):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "clNetData_plans_2D"
        self.plans_fname = join(self.preprocessed_output_folder, "clNetPlans_plans_2D.json")
        self.unet_base_num_features = 32

    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):

        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)

        dataset_num_voxels = np.prod(new_median_shape, dtype=np.int64) * num_cases
        input_patch_size = new_median_shape[1:]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)

        # we pretend to use 30 feature maps. This will yield the same configuration as in V1. The larger memory
        # footpring of 32 vs 30 is mor ethan offset by the fp16 training. We make fp16 training default
        # Reason for 32 vs 30 feature maps is that 32 is faster in fp16 training (because multiple of 8)
        ref = Generic_UNet.use_this_for_batch_size_computation_2D * Generic_UNet.DEFAULT_BATCH_SIZE_2D / 2
        # for batch size 2
        here = Generic_UNet.compute_approx_vram_consumption(new_shp,
                                                            network_num_pool_per_axis,
                                                            30,
                                                            self.unet_max_num_filters,
                                                            num_modalities, num_classes,
                                                            pool_op_kernel_sizes,
                                                            conv_per_stage=self.conv_per_stage)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape[1:])[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(current_spacing[1:], tmp, self.unet_featuremap_min_edge_length,
                                        self.unet_max_numpool)
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], new_shp,
                                                                 self.unet_featuremap_min_edge_length,
                                                                 self.unet_max_numpool)

            here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                self.unet_base_num_features,
                                                                self.unet_max_num_filters, num_modalities,
                                                                num_classes, pool_op_kernel_sizes,
                                                                conv_per_stage=self.conv_per_stage)
            # print(new_shp)

        batch_size = int(np.floor(ref / here) * 2)
        input_patch_size = new_shp

        if batch_size < self.unet_min_batch_size:
            raise RuntimeError("This should not happen")

        # check if batch size is too large (more than 5 % of dataset)
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        batch_size = max(1, min(batch_size, max_batch_size))

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'do_dummy_2D_data_aug': False
        }
        return plan

    def plan_experiment(self):
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero maks for normalizaion?", use_nonzero_mask_for_normalization)

        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']
        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        if self.dataset_properties['fixed_resolution'] is not None:
            target_spacing = np.array(self.dataset_properties['fixed_resolution'], dtype=float)
            print("We are using the fixed resolution from 'dataset.json' file", target_spacing)
        else:
            target_spacing = self.get_target_spacing()
        new_shapes = np.array([np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)])

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("we don't want feature maps smaller than ", self.unet_featuremap_min_edge_length, " in the bottleneck")

        # how many stages will the image pyramid have?
        self.plans_per_stage = []

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("the transposed median shape of the dataset is ", median_shape_transposed)

        if self.dataset_properties['fixed_pool'] is not None:
            fixed_pool = self.dataset_properties['fixed_pool']
            fixed_network_num_pool_per_axis = [len(fixed_pool), len(fixed_pool), len(fixed_pool)]
            for i_pool in range(len(fixed_pool)):
                for j_pool in range(len(fixed_pool[i_pool])):
                    if fixed_pool[i_pool][j_pool] < 2:
                        fixed_network_num_pool_per_axis[i_pool] -= 1
            print("We are using the fixed Pool Op Kernel Sizes from 'dataset.json' file", fixed_pool)
        else:
            fixed_pool = None
            fixed_network_num_pool_per_axis = None
        if self.dataset_properties['fixed_conv'] is not None:
            fixed_conv = self.dataset_properties['fixed_conv']
            print("We are using the fixed Conv Op Kernel Sizes from 'dataset.json' file", fixed_conv)
        else:
            fixed_conv = None
        if self.dataset_properties['fixed_patch_size'] is not None:
            fixed_patch_size = np.array(self.dataset_properties['fixed_patch_size'], dtype=int)
            print("We are using the fixed Patch Size from 'dataset.json' file", fixed_patch_size)
        else:
            fixed_patch_size = None
        if self.dataset_properties['fixed_batch_size'] is not None:
            fixed_batch_size = int(np.round(self.dataset_properties['fixed_batch_size']))
            print("We are using the fixed Batch Size from 'dataset.json' file", fixed_batch_size)
        else:
            fixed_batch_size = None
        fullres_plan = self.get_properties_for_stage(target_spacing_transposed,
                                                     target_spacing_transposed,
                                                     median_shape_transposed,
                                                     len(self.list_of_cropped_npz_files),
                                                     num_modalities, len(all_classes) + 1)
        if fixed_pool is not None and fixed_conv is not None and \
                fixed_patch_size is not None and fixed_batch_size is not None:
            fixed_do_dummy_2D_data_aug = (max(fixed_patch_size) / fixed_patch_size[0]) > self.anisotropy_threshold
            fullres_plan["batch_size"] = fixed_batch_size
            fullres_plan["patch_size"] = fixed_patch_size
            fullres_plan["num_pool_per_axis"] = fixed_network_num_pool_per_axis
            fullres_plan["do_dummy_2D_data_aug"] = fixed_do_dummy_2D_data_aug
            fullres_plan["pool_op_kernel_sizes"] = fixed_pool
            fullres_plan["conv_kernel_sizes"] = fixed_conv
        self.plans_per_stage.append(fullres_plan)

        print(self.plans_per_stage)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        normalization_schemes = self.determine_normalization_scheme()
        # deprecated
        only_keep_largest_connected_component, min_size_per_class, min_region_size_per_class = None, None, None

        # these are independent of the stage
        plans = {'num_stages': len(list(self.plans_per_stage.keys())), 'num_modalities': num_modalities,
                 'modalities': modalities, 'normalization_schemes': normalization_schemes,
                 'dataset_properties': self.dataset_properties, 'list_of_npz_files': self.list_of_cropped_npz_files,
                 'original_spacings': spacings, 'original_sizes': sizes,
                 'preprocessed_data_folder': self.preprocessed_output_folder, 'num_classes': len(all_classes),
                 'all_classes': all_classes, 'base_num_features': self.unet_base_num_features,
                 'use_mask_for_norm': use_nonzero_mask_for_normalization,
                 'keep_only_largest_region': only_keep_largest_connected_component,
                 'min_region_size_per_class': min_region_size_per_class, 'min_size_per_class': min_size_per_class,
                 'transpose_forward': self.transpose_forward, 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage,
                 'preprocessor_name': self.preprocessor_name,
                 }

        self.plans = plans
        self.save_my_plans()
