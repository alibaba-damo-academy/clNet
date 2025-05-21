import glob

import torch.cuda
import torch.multiprocessing as mp
import torch.nn.functional as F

from clnet.inference.utils import *
from clnet.network_architecture.custom_modules.pruning_modules import *
from clnet.preprocessing.preprocessing import resample_data_or_seg, get_do_separate_z, get_lowres_axis

from batchgenerators.utilities.file_and_folder_operations import *


def resample_seg(filenames_nii: list, filenames_npy: list, filenames_json: list,
                 save_npz: bool = True, mode: str = "normal", verbose: bool = False):
    for filename_npy, filename_nii, filename_json in zip(filenames_npy, filenames_nii, filenames_json):
        if verbose:
            print("Resampling Seg", filename_nii)
        seg_npy = np.load(filename_npy)
        header = json.load(open(filename_json))
        original_size_of_raw_data_cropped = header["original_size_of_raw_data_cropped"]
        original_size_of_raw_data = header["original_size_of_raw_data"]
        itk_origin = header["itk_origin"]
        itk_spacing = header["itk_spacing"]
        itk_direction = header["itk_direction"]
        resampled_spacing = header["size_after_cropping"]
        if mode == "normal":
            if get_do_separate_z(itk_spacing):
                do_separate_z = True
                axis = get_lowres_axis(itk_spacing)
            elif get_do_separate_z(resampled_spacing):
                do_separate_z = True
                axis = get_lowres_axis(resampled_spacing)
            else:
                do_separate_z = False
                axis = None
            seg_cropped = resample_data_or_seg(seg_npy, original_size_of_raw_data_cropped, True, axis, do_separate_z)
        else:
            seg_tensor = torch.from_numpy(np.array(seg_npy, dtype=np.float32))
            seg_tensor = F.interpolate(seg_tensor[None][None], original_size_of_raw_data_cropped, mode="nearest")[0][0]
            seg_cropped = seg_tensor.detach().cpu().numpy().astype(np.int16)

        seg_dat = np.zeros(original_size_of_raw_data, dtype=seg_cropped.dtype)
        bbox = header["crop_bbox"]
        seg_dat[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = seg_cropped
        seg_sitk = sitk.GetImageFromArray(seg_dat)
        seg_sitk.SetOrigin(itk_origin)
        seg_sitk.SetSpacing(itk_spacing)
        seg_sitk.SetDirection(itk_direction)
        sitk.WriteImage(seg_sitk, filename_nii)
        if not save_npz:
            if os.path.isfile(filename_npy):
                os.remove(filename_npy)
            if os.path.isfile(filename_json):
                os.remove(filename_json)


def postprocess_resample_seg(trainer_heads_summarized: dict, output_folder: str, case_ids: np.ndarray, decoder_or_support: str,
                             overwrite_existing_pred: bool, num_threads_nifti_save: int, save_intermediate_result_for_debug: bool,
                             verbose: bool = False):
    all_decoders = list(trainer_heads_summarized[decoder_or_support].keys())
    filenames_nii, filenames_npy, filenames_json = [], [], []
    for case_id in case_ids:
        # try to locate predictions
        for decoder in all_decoders:
            filename_nii = os.path.join(output_folder, case_id, case_id + "_{}_{}.nii.gz".format(decoder, decoder_or_support))
            filename_npy = os.path.join(output_folder, case_id, case_id + "_{}_{}.npy".format(decoder, decoder_or_support))
            filename_json = os.path.join(output_folder, case_id, case_id + "_{}_{}.json".format(decoder, decoder_or_support))
            if os.path.isfile(filename_npy) and os.path.isfile(filename_json):
                if overwrite_existing_pred or not os.path.isfile(filename_nii):
                    filenames_nii.append(filename_nii)
                    filenames_npy.append(filename_npy)
                    filenames_json.append(filename_json)
    num_process = min(len(filenames_nii), num_threads_nifti_save)
    # resample the segmentations back to their original shape
    processes_gpu = []
    for i in range(num_process):
        p_gpu = mp.Process(target=resample_seg,
                           args=(filenames_nii[i::num_process], filenames_npy[i::num_process], filenames_json[i::num_process],
                                 save_intermediate_result_for_debug, verbose))
        p_gpu.start()
        processes_gpu.append(p_gpu)
    for p_gpu in processes_gpu:
        p_gpu.join()
    # # try to remove the remaining intermediate files
    # if not save_intermediate_result_for_debug:
    #     for case_id in case_ids:
    #         # try to locate the remaining intermediate files
    #         remaining_bm_files = glob.glob(os.path.join(output_folder, case_id, case_id + "*BodyMask*.nii.gz"))
    #         remaining_npy_files = glob.glob(os.path.join(output_folder, case_id, case_id + "*.npy"))
    #         # remaining_json_files = glob.glob(os.path.join(output_folder, case_id, case_id + "*.json"))
    #         for remaining_bm_file in remaining_bm_files:
    #             if os.path.isfile(remaining_bm_file):
    #                 os.remove(remaining_bm_file)
    #         for remaining_npy_file in remaining_npy_files:
    #             if os.path.isfile(remaining_npy_file):
    #                 os.remove(remaining_npy_file)
    #         # for remaining_json_file in remaining_json_files:
    #         #     if os.path.isfile(remaining_json_file):
    #         #         os.remove(remaining_json_file)


def dump_to_nii(seg_npy, filenames_nii, header):
    itk_origin = header["itk_origin"]
    itk_spacing = header["itk_spacing"]
    itk_direction = header["itk_direction"]
    for i, filename_nii in enumerate(filenames_nii):
        seg_npy_current = seg_npy[i].astype(np.uint16)
        seg_sitk = sitk.GetImageFromArray(seg_npy_current)
        seg_sitk.SetOrigin(itk_origin)
        seg_sitk.SetSpacing(itk_spacing)
        seg_sitk.SetDirection(itk_direction)
        sitk.WriteImage(seg_sitk, filename_nii)


def process_on_gpu(device, seg_npy, original_size_of_raw_data, filenames_nii, header, num_threads_nifti_save):
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    seg_tensor = torch.from_numpy(np.array(seg_npy, dtype=np.float16)).to(device)
    seg_tensor = F.interpolate(seg_tensor[None], size=original_size_of_raw_data, mode="nearest")[0]
    seg_tensor = seg_tensor.detach().cpu().numpy()
    num_process = min(len(filenames_nii), num_threads_nifti_save)
    processes_cpu = []
    for i in range(num_process):
        p_cpu = mp.Process(target=dump_to_nii, args=(seg_tensor[i::num_process], filenames_nii[i::num_process], header))
        p_cpu.start()
        processes_cpu.append(p_cpu)
    for p_cpu in processes_cpu:
        p_cpu.join()


def resample_seg_on_gpu(trainer_heads_summarized, output_folder, case_ids, decoder_or_support,
                        overwrite_existing_pred, num_threads_nifti_save):
    """
    DO NOT USE IT
    Resample on GPU works is inefficient: 1) limited number of GPUs, 2) CPU - GPU bios overhead.
    """
    all_decoders = list(trainer_heads_summarized[decoder_or_support].keys())
    for case_id in case_ids:
        filenames_nii, filenames_npy, filenames_json = [], [], []
        # load the header file stored in json.
        filename_json = join(output_folder, case_id, case_id + ".json")
        # try to locate predictions
        for decoder in all_decoders:
            filename_nii = join(output_folder, case_id, case_id + "_{}_{}.nii.gz".format(decoder, decoder_or_support))
            filename_npy = join(output_folder, case_id, case_id + "_{}_{}.npy".format(decoder, decoder_or_support))
            if os.path.isfile(filename_npy) and os.path.isfile(filename_json):
                if overwrite_existing_pred or not os.path.isfile(filename_nii):
                    filenames_nii.append(filename_nii)
                    filenames_npy.append(filename_npy)
                    filenames_json.append(filename_json)
        header = json.load(open(filename_json))
        original_size_of_raw_data = header["original_size_of_raw_data"]

        pred_npy = np.vstack([[np.load(filename_npy) for filename_npy in filenames_npy]])
        # check how many predictions (half precision) can a 16GB GPU hold.
        # By default, it should hold 2**33 = (16*1024*1024*1024)/2 half precision numbers.
        # However, concerning calculation overheads, GPU overhead, and bios preserve, we choose 2**30 to be safety.
        resampled_size_of_raw_data = pred_npy.shape[1:]
        max_size = [max(resampled_size_of_raw_data[i], original_size_of_raw_data[i]) for i in range(3)]
        max_num_of_predictions_stacked_for_16gb_gpu = min(pred_npy.shape[0], int((2 ** 30) / np.prod(max_size)))

        num_sections = min(torch.cuda.device_count(), max(1, int(pred_npy.shape[0] / max_num_of_predictions_stacked_for_16gb_gpu)))
        devices = [torch.device(f'cuda:{i}') for i in range(num_sections)]
        num_threads_nifti_save_per_gpu = max(1, int(num_threads_nifti_save / num_sections))
        processes_gpu = []
        for i in range(num_sections):
            p_gpu = mp.Process(target=process_on_gpu, args=(devices[i], pred_npy[i::num_sections], original_size_of_raw_data,
                                                            filenames_nii[i::num_sections], header, num_threads_nifti_save_per_gpu))
            p_gpu.start()
            processes_gpu.append(p_gpu)
        for p_gpu in processes_gpu:
            p_gpu.join()
