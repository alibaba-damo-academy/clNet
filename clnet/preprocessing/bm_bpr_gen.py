import os.path

import torch.multiprocessing as mp
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, iterate_structure

from clnet.inference.utils import *
from clnet.bpreg.bpr_pred import bpr_gen
from clnet.configuration import default_num_threads


def bm_bpr_gen(input_folder: str, output_folder: str, overwrite_existing: bool = False, num_threads_preprocessing: int = default_num_threads,
               disable_bm_bpr=False, skip_bm: bool = False, verbose: bool = False):
    # check input folder integrity
    case_ids = check_input_folder_and_return_case_ids(input_folder, expected_num_modalities=1)
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    image_lists = [[join(input_folder, filename) for filename in all_files
                    if filename[:len(case_id)].startswith(case_id) and len(filename) == (len(case_id) + 12)] for case_id in case_ids]
    if not disable_bm_bpr:
        # output check
        process_list = preload_img(case_ids, output_folder, image_lists, overwrite_existing, verbose)

        # generate coarse body mask
        if not verbose:
            print("Generating body masks")
        bm_list = [join(output_folder, case_id, case_id + "_BodyMask.nii.gz") for case_id in case_ids]
        num_process = min(len(bm_list), num_threads_preprocessing)
        if len(bm_list) > 0:
            processes = []
            for i in range(num_process):
                p = mp.Process(target=bm_coarse_gen,
                               args=(bm_list[i::num_process], process_list[i::num_process], overwrite_existing, skip_bm, verbose))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        # generate bpr & clean up bpr scores
        if not verbose:
            print("Generating bpr scores")
        bpr_output_list = [join(output_folder, i, i + "_bpr.json") for i in case_ids]
        bpr_input_filename_list = [join(output_folder, i, i + ".npy") for i in case_ids]
        bpr_gen(bpr_input_filename_list, bpr_output_list, overwrite_existing, verbose)
        bpr_output_for_clean = [[bpr_file, overwrite_existing, verbose] for bpr_file in bpr_output_list]
        if num_threads_preprocessing > 2:
            with mp.Pool(int(num_threads_preprocessing)) as p:
                p.map(bpr_score_clean, bpr_output_for_clean)
        else:
            for bpr_file in bpr_output_for_clean:
                bpr_score_clean(bpr_file)

        # final check list
        cleaned_image_lists, cleaned_bpr_output_list, cleaned_bm_list = [], [], []
        for case_id in case_ids:
            check_input_image = join(input_folder, case_id + "_0000.nii.gz")
            check_output_bpr = join(output_folder, case_id, case_id + "_bpr.json")
            check_output_bm = join(output_folder, case_id, case_id + "_BodyMask.nii.gz")
            if [check_input_image] in image_lists and os.path.isfile(check_output_bm) and os.path.isfile(check_output_bpr):
                cleaned_image_lists.append([check_input_image])
                cleaned_bpr_output_list.append(check_output_bpr[:-len(".json")] + "_cleaned.json")
                cleaned_bm_list.append(check_output_bm)
    else:
        cleaned_image_lists, cleaned_bpr_output_list, cleaned_bm_list = [], None, None
        for case_id in case_ids:
            check_input_image = join(input_folder, case_id + "_0000.nii.gz")
            if [check_input_image] in image_lists:
                cleaned_image_lists.append([check_input_image])

    return cleaned_image_lists, cleaned_bpr_output_list, cleaned_bm_list


def preload_img(case_ids: list, output_folder: str, input_image_lists: list, overwrite_existing: bool = True, verbose: bool = True):
    maybe_mkdir_p(output_folder)
    for case_id in case_ids:
        os.makedirs(join(output_folder, case_id), exist_ok=True)

    output_lists = [join(output_folder, case_id, case_id) for i, case_id in enumerate(case_ids)]
    for input_image_list, output_list in zip(input_image_lists, output_lists):
        if verbose:
            print("Preloading images", output_list)
        output_file_header = output_list + ".json"
        output_file_data = output_list + ".npy"

        if not os.path.isfile(output_file_header) or not os.path.isfile(output_file_data) or overwrite_existing:
            img_original = [sitk.ReadImage(f) for f in input_image_list]
            dat_original = np.vstack([sitk.GetArrayFromImage(img)[None] for img in img_original])
            properties = OrderedDict()
            properties["itk_origin"] = img_original[0].GetOrigin()
            properties["itk_spacing"] = img_original[0].GetSpacing()
            properties["itk_direction"] = img_original[0].GetDirection()
            # dump properties to json
            with open(output_file_header, "w") as f:
                json.dump(properties, f)
            np.save(output_file_data, dat_original)
    return output_lists


def bm_coarse_gen(filename_bm_list: list, filename_ct_list: list, overwrite: bool = True,
                  skip_bm: bool = False, verbose: bool = True, ct_bkg_intensity_threshold: int = -200):
    for filename_bm, filename_ct in zip(filename_bm_list, filename_ct_list):
        if not os.path.isfile(filename_bm) or overwrite:
            if verbose:
                print("Generating coarse body mask", filename_bm)
            try:
                img_header = json.load(open(filename_ct + ".json"))
                dat = np.load(filename_ct + ".npy")[0]
                img_origin = img_header["itk_origin"]
                img_direction = img_header["itk_direction"]
                img_spacing = img_header["itk_spacing"]
            except:
                img = sitk.ReadImage(filename_ct)
                dat = sitk.GetArrayFromImage(img)
                img_origin = img.GetOrigin()
                img_direction = img.GetDirection()
                img_spacing = img.GetSpacing()

            struct_c8_r3 = generate_binary_structure(2, 2)
            struct_c8_r7 = iterate_structure(struct_c8_r3, 3)
            # 2d thresholding, erosion, dilation, and cc analysis and fill 2d hole
            if skip_bm:
                mask_3d_data = np.ones(dat.shape, dtype=np.int8)
            else:
                mask_3d_data = np.zeros(dat.shape, dtype=np.int8)
                cc_filter = sitk.ConnectedComponentImageFilter()
                relabel_filter = sitk.RelabelComponentImageFilter()
                for ind_z in range(0, dat.shape[0]):
                    ct_2d_data = dat[ind_z, :, :]
                    mask_2d_data = ct_2d_data > ct_bkg_intensity_threshold
                    # 2d opening
                    mask_2d_data = binary_erosion(mask_2d_data, structure=struct_c8_r7, iterations=1)
                    mask_2d_data = binary_dilation(mask_2d_data, structure=struct_c8_r7, iterations=1)
                    # fill 2d hole and remove small disconnected component
                    mask_2d_img = sitk.GetImageFromArray(mask_2d_data.astype(np.int8))
                    cc_filter.SetFullyConnected(False)
                    obj_label = cc_filter.Execute(mask_2d_img)
                    relabel_filter.SortByObjectSizeOn()
                    obj_relabel = relabel_filter.Execute(obj_label)
                    relabel_data = sitk.GetArrayFromImage(obj_relabel)
                    relabel_data[relabel_data > 1] = 0
                    mask_3d_data[ind_z, :, :] = relabel_data
                    mask_3d_data[ind_z, :, :] = binary_fill_holes(relabel_data).astype(relabel_data.dtype)

            # Try to get rid of small outliers
            img_foreground_mask = sitk.GetImageFromArray(mask_3d_data)
            img_foreground_mask.SetOrigin(img_origin)
            img_foreground_mask.SetDirection(img_direction)
            img_foreground_mask.SetSpacing(img_spacing)
            sitk.WriteImage(img_foreground_mask, filename_bm)


def bpr_score_clean(input_list):
    bpr_file, overwrite, verbose = input_list
    if verbose:
        print("Cleaning bpr scores", bpr_file)

    output_bpr_file = bpr_file[:-len(".json")] + "_cleaned.json"
    if not os.path.isfile(output_bpr_file) or overwrite:
        bpr_json_raw = json.load(open(bpr_file))
        bpr_scores = np.asarray(bpr_json_raw["cleaned slice scores"])
        mean_interval = np.nanmean(np.diff(bpr_scores))
        for i, s in enumerate(bpr_scores):
            if i == 0:
                continue
            if (np.isnan(s)) and (not np.isnan(bpr_scores[i - 1])):
                bpr_scores[i] = bpr_scores[i - 1] + mean_interval

        for i, s in enumerate(bpr_scores[::-1]):
            i = len(bpr_scores) - 1 - i
            if (np.isnan(s)) and (not np.isnan(bpr_scores[i + 1])):
                bpr_scores[i] = bpr_scores[i + 1] - mean_interval
        bpr_json_raw["cleaned slice scores"] = list(bpr_scores)
        save_json(bpr_json_raw, output_bpr_file, sort_keys=False)


if __name__ == "__main__":
    input_folder_ = "/nas/dazhou.guo/Data_Partial/clNet_raw_data/Task016_StructSeg_OAR22/imagesTs"
    output_folder_ = "/nas/dazhou.guo/Data_Partial/clNet_raw_data/Task016_StructSeg_OAR22/predsTs_debug"
    mp.set_start_method("spawn", force=True)
    bm_bpr_gen(input_folder_, output_folder_, True, verbose=True)
