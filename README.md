

# Unified, Accurate, Generalizable and Non-forgetting Continual Segmentation Models of Fine-grained Whole-body Organs, Lymph Node Stations and Lesions in 3D CT Scans

***
**This work is based on the [nnUNet-v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).**
**Great thanks to Fabian et. al.**
**Please cite [nnUNet](https://www.nature.com/articles/s41592-020-01008-z) paper and read its [ReadMe](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/readme.md) for reference.**
***

## What is Continual Semantic Segmentation?
In real clinical environments,  continual semantic segmentation is often preferred, 
as it allows segmentation models to dynamically adapt to new organs or tumors 
without requiring access to previous training datasets.

One of the main reasons for this preference is the difficulty of re-accessing previous datasets, 
largely due to strict patient privacy regulations, such as HIPAA (Health Insurance Portability and Accountability Act), 
and the considerable challenges of securely storing and managing large volumes of medical data over time

***
## What is CL-Net?
In this repo, we propose a novel architecture-based continual learning network (clNet) for multi-organ segmentation. On the basis of
the common encoder + decoder architecture of segmentation networks, we demonstrate that its encoder is capable of
extracting representative deep features (non-specific to organ or body part) for the new data. Hence, we can freeze
the encoder and incrementally add a separate decoder for each new learning task.

![CL-Net](documents/fig1_v5.png)

Upon release, clNet was evaluated on both public and private CT datasets, ranging from various body parts and different anatomical structures. 
Despite competing with handcrafted solutions for each target, clNet's fully automated pipeline achieved the leading performance on all tasks. 


## What does clNet include?
1) General encoder training / loading
2) Body part parsing
3) EMA-enable model training / loading
4) Feature-level supporting 
5) "Lottery ticket"-based auto-decoding path pruning
6) Prediction Merging

Upon release, please read and cite the [following ICCV2023 paper](https://arxiv.org/abs/2302.00162): 

    Ji, Zhanghexuan*, Dazhou Guo*, Puyang Wang, Ke Yan, Jia Ge, Xianghua Ye, Minfeng Xu, Dakai Jin.
    "Continual segment: Towards a single, unified and accessible continual segmentation model of 143 whole-body organs in ct scans." 
    In Proceedings of the IEEE International Conference on Computer Vision. 2023
***
## Installation
Segment in the Wild has been tested on Linux and requires a GPU. For inference, a single GPU with a minimum of 4 GB VRAM is necessary. 
For training clNet models, the GPU should have at least 12 GB of VRAM. FYI, we recommend pairing a strong CPU with the GPU. 
Preferably, 8 CPU cores (16 threads) are recommended.

For use as integrative **framework**:
- python >= 3.10
- CUDA >= 11.7
- GCC >= 10.2.1
- PyTorch >= 2.0.1
```bash
pip install -U setuptools
git clone git@gitlab.alibaba-inc.com:med-rt/clNet.git
cd clNet
pip install -e .
```

The requirements can be found in `setup.py` file. 

***
## Data Preparation And Folder Structure
### Set clNet Root Data Path 
Similar to nnUNet-V1's path settings, clNet requires you to set the root data path as an environment variable named `clNet_raw_data_base`. This root path should include the raw data folder `clNet_raw_data`. The folders `clNet_cropped_data`, `preprocessed`, and `results/clNet` will be generated automatically under this root path. For detailed path settings, please refer to the paths.py Python file. For detailed path settings, please refer to the `paths.py` Python file. 

You can set the `clNet_raw_data_base` environment variable by adding the following line to your `.bashrc` or `.bash_profile` (if you are using Bash): 
```bash
export clNet_raw_data_base=<your clNet root data path>
```

### Setup Pre-trained Models
**_Pre-trained models_** follows the nnUNet-v1's model structure and must be located in the `results` folder 
(which you either define when installing clNet, setup `clnet.path.py`).

### Data Re-orientation 
Data re-orientation is essential to ensure all images are aligned with the same orientation as the pre-trained General Encoder. 
Please find the Python file in `data_prep` folder that converts the input image to ***LPS*** orientation.

    clnet/
    ├── data_prep
        ├── Step0_ConvertOrientation.py

### Setup Training Datasets
**_Tasks/Datasets_** follows the nnUNet-V1's dataset structure and must be located in the `clNet_raw_data` folder 
(which you either define when installing clNet, setup `clnet.path.py`). Each segmentation task/dataset is stored as a separate 'Task'. 
Tasks are associated with a dataset ID, a three/four digit integer, and a task name (which you can freely choose): 
For example, `Task011_HNOAR` has `HNOAR` as task name and the task id is `011`. Tasks are stored in the `clNet_raw_data` folder like this:


    clNet_raw_data/
    ├── Task001_GE
    ├── Task002_AbdomenCT1K_Organ12
    ├── Task006_Chest_EsoCancer_Organ35
    ├── Task011_HNOAR_42
    ├── Task500_TotalSegV2
    ├── ...

Within each dataset folder, the following structure is expected:

    Dataset001_BrainTumour/
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs  # optional
    └── labelsTr

- **imagesTr** contains the images belonging to the training cases. nnU-Net will perform pipeline configuration, training with 
cross-validation, as well as finding postprocessing and the best ensemble using this data. 
- **imagesTs** (optional) contains the images that belong to the test cases. clNet does not use them! 
- **labelsTr** contains the images with the ground truth segmentation maps for the training cases. 
- **dataset.json** contains metadata of the dataset.
- Note that the imagesTs folder is optional and does not have to be present.

For example, the following is the structure of the `Task001_GE` dataset:

    Task001_GE/
    ├── dataset.json
    ├── imagesTr
    │   ├── GE_001_0000.nii.gz
    │   ├── GE_002_0000.nii.gz
    │   ├── GE_003_0000.nii.gz
    │   ├── GE_004_0000.nii.gz
    │   ├── GE_005_0000.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── GE_500_0000.nii.gz
    │   ├── GE_501_0000.nii.gz
    │   ├── GE_502_0000.nii.gz
    │   ├── GE_503_0000.nii.gz
    │   ├── GE_504_0000.nii.gz
    │   ├── ...
    └── labelsTr
        ├── GE_001.nii.gz
        ├── GE_002.nii.gz
        ├── ...

To familiarize yourself of the framework, we recommend you to go through the examples regarding **data preparation** and
**data preprocessing** in [nnUNet v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). 

***

## Training
### Body Part Regression (bpreg) pretrained model
The *_bpreg_* pretrained model can be automatically downloaded from its official [link](https://zenodo.org/records/5113483#.YPaBkNaxWEA).

If the link is broken or the model is unavailable, please download it manually from the following  link:
```text
- Alibaba Cloud Object Storage Service - 
oss://med-rt/DazhouGuo/Data_nas/Data_Partial/results/public_bpr_model/
```

###  General Encoder pretrained weights
The clNet requires the General Encoder for general feature extraction. The pre-trained General Encoder can be found at:
```text
- Alibaba Cloud Object Storage Service - 
oss://med-rt/DazhouGuo/Data_nas/Data_Partial/results/clNet/3d_fullres/Task001_GeneralEncoder/
```

After finish preparing the training data 
```bash
clNet_plan_and_preprocess -t xxx
```

###  Training configuration JSON example
#### Note if the training head targets `GTV/tumor`, please identify `GTV` in the decoding head naming!

```json
{
  "clnet_general_encoder": "GeneralEncoder",
  "GeneralEncoder": {
    "task": "Task001_GeneralEncoder",
    "fold": "all",
    "load_only_encoder": true
  },
  
  "Task_1": {
    "train_order": 1,
    "task": "Task002_ExampleDataset",
    "fold": 0,
    "continue_training": false,
    "decoders": {
      "BrainStem": 1,
      "Eyes": [2, 3],
      "Lens": "4, 5",
      "NPC_GTV": 6
    },
    "supporting": {
      "lens": ["Eyes"],
      "NPC_GTV": ["BrainStem", "Eyes"]
    },
    "decoders_to_train": [
      "all"
    ]
  },
  
  "Task_2": {
    "task": "Task003_ExampleDataset",
    "fold": 0,
    "decoders": {
      "Ribs_left": "1-12",
      "Ribs_right": "13-24"
    },
    "decoders_to_train": [
      "all"
    ]
  }
}
```


#### [Complete configuration JSON setup for Segment in the Wild](documents/CompleteJSON.md)

### Training ###

- Single GPU Training
```bash
clNet_train `pth_to_train_json_file`
```
- Multiple GPUs DDP training
```bash
torchrun --nproc_per_node=`num_of_GPUs` clnet/run/run_training_ddp.py `pth_to_train_ge_json_file`
```

### Warmup, EMA, and Pruning ###
We enabled network Warmup, EMA, and Pruning for the training process by default. 
#### Warmup ####
- The network warmup epoch is set to `5` by default. 
- If the warmup epoch is set to `0`, then the network warmup will be disabled.
- If the warmup epoch is set to a floating value between `0` and `1`, then the network warmup epoch will be set to `int(warmup_epoch * num_of_epochs)`.
#### EMA Updating #### 
- If the pruning is set to `True`, then the EMA updating will start after the pruning is completed. 
- If the pruning is set to `False`, then the EMA updating will start after the first training epoch. 
#### Pruning ####
- Each decoder is progressively pruned. The progressive pruning ratios are set to `[80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 99, 99.2, 99.4, 99.6, 99.8]`
- The performance of the pruned model is evaluated based on `0.5% - 99.5%` of the validation DSCs. 
- The allowed performance drop (caused by pruning) is limited to a maximum of `2%` in DSC. 
If the evaluation performance decreases by more than `2%`, the decoder will be re-pruned, with a maximum of `8` attempts, 
after which the pruning will be stopped.
- After pruning, the decoding head is trained for an additional `50` epochs to recover potential performance drop. 
- Pruning is ***NOT*** performed on the Encoder.

### Hyperparameters ###
The hyperparameters are listed in `configuration.py` Python file. 
***

## Inference ###
We enabled clNet for multi-GPU inference, with each GPU targeting a separate decoding head. For example, if you have `4` GPUs and `8` decoding heads, 
then the `GPU-0` will predict the `1st & 2nd` decoding heads, `GPU-1` will predict the `3rd & 4th` decoding heads, and so on. 
```bash
clNet_pred `pth_to_inference_json_file` -i `pth_to_input_folder` -o `pth_to_output_folder`
```