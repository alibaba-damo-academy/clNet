import numpy as np
import SimpleITK as sitk
from copy import deepcopy
from collections import OrderedDict
from torch.multiprocessing import Process, Queue

from batchgenerators.utilities.file_and_folder_operations import *

from clnet.preprocessing.preprocessing import GenericPreprocessor


def preprocess_save_to_queue(preprocess_fn, plan, q, list_of_lists, output_files, bm_files, bpr_range,
                             header_json_filenames, output_filenames_npy):
    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            header_json_file = header_json_filenames[i]
            output_file_npy = output_filenames_npy[i]
            # print("preprocessing", output_file)

            d, _, properties = preprocess_fn(plan, l, bm_files[i], bpr_range[i])
            """There is a problem with python process communication that prevents us from communicating objects 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            # print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be saved, 4 because float32 is 4 bytes
                print("This output is too large for python process-process communication. Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, properties, header_json_file, output_file_npy)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    # else:
    #     print("This worker has ended successfully, no errors to report")


def preprocess_multithreaded(plan, list_of_lists, output_files, num_processes=2, header_json_filenames=None,
                             output_filenames_npy=None, bm_files=None, bpr_range=None):
    if bm_files is None or len(bm_files) == 0:
        bm_files = [None] * len(list_of_lists)

    if bpr_range is None or len(bpr_range) == 0:
        bpr_range = [None] * len(list_of_lists)

    if header_json_filenames is None or len(header_json_filenames) == 0:
        header_json_filenames = [None] * len(list_of_lists)

    if output_filenames_npy is None or len(output_filenames_npy) == 0:
        output_filenames_npy = [None] * len(list_of_lists)

    # try to use as few processes as possible
    num_processes = min(len(list_of_lists), num_processes)

    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue,
                     args=(preprocess_patient, plan, q, list_of_lists[i::num_processes],
                           output_files[i::num_processes], bm_files[i::num_processes], bpr_range[i::num_processes],
                           header_json_filenames[i::num_processes], output_filenames_npy[i::num_processes]))
        pr.start()
        processes.append(pr)
    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # this should not happen but better safe than sorry right
            p.join()

        q.close()


def preprocess_patient(plan, input_files, bm_files=None, bpr_range=None):
    normalization_schemes = plan["normalization_schemes"]
    use_mask_for_norm = plan["use_mask_for_norm"]
    transpose_forward = plan["transpose_forward"]
    intensity_properties = plan["dataset_properties"]["intensityproperties"]
    target_resolution = plan["plans_per_stage"][0]["current_spacing"]
    patch_size = plan["plans_per_stage"][0]["patch_size"]
    preprocessor = GenericPreprocessor(normalization_schemes, use_mask_for_norm, transpose_forward, intensity_properties)
    d, s, properties = preprocessor.preprocess_test_case(input_files, target_resolution, bm_files, bpr_range, patch_size=patch_size)
    return d, s, properties


def check_input_folder_and_return_case_ids(input_folder, expected_num_modalities=1):
    # print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    maybe_case_ids = np.unique([i[:-12] for i in files])
    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def get_largest_component(pred, fully_connected=True):
    pred_data = sitk.GetArrayFromImage(pred)
    pred_labeled = sitk.ConnectedComponent(pred, fully_connected)
    pred_labeled_data = sitk.GetArrayFromImage(pred_labeled)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(pred_labeled)

    maxlabel = 0
    maxsize = 0

    for label in stats.GetLabels():
        size = stats.GetPhysicalSize(label)
        if maxsize < size:
            maxlabel = label
            maxsize = size

    pred_data[pred_labeled_data != maxlabel] = 0
    pred_data[pred_labeled_data == maxlabel] = 1

    pred_master = sitk.GetImageFromArray(pred_data)
    pred_master.CopyInformation(pred)

    return pred_master


def per_slice_fill_holes(mask):
    for slice_idx in range(mask.GetSize()[-1]):
        mask[:, :, slice_idx] = sitk.BinaryFillhole(mask[:, :, slice_idx])
    return mask


def load_case_from_list_of_files(data_files, seg_file=None, bpr_range=None, half_precision=False):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()
    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None]
        if bpr_range is not None:
            seg_npy[:, :bpr_range[0]] = 0
            seg_npy[:, bpr_range[1]:] = 0
    else:
        seg_npy = None
    if half_precision:
        return data_npy.astype(np.float16), seg_npy.astype(np.float16), properties
    else:
        return data_npy.astype(np.float32), seg_npy.astype(np.float32), properties
