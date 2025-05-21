###  General Encoder training / loading
The clNet requires the General Encoder for general feature extraction. The pre-trained General Encoder can be found at:
```text
oss://med-rt/DazhouGuo/Task001_SuperGeneralEncoder
```
To train the general encoder please refer to the following the example "train_ge.json".
```bash
CUDA_VISIBLE_DEVICES=0 clNet_train `pth_to_train_ge_json_file`
```
Complete Configuration JSON example:
```json
{
  "clnet_enable_ddp": true,
  "clnet_general_encoder": "GeneralEncoder",
  "GeneralEncoder": {
    "task": "Task001_SuperGeneralEncoder",
    "fold": "all",
    "no_mirroring": true,
    "continue_training": false,
    "decoder_update": false,
    "load_only_encoder": false,
    "full_network": false,
    "pretrain_model_name": "final",
    "optimizer": "adamw",
    "amsgrad": true,
    "encoder_architecture_setup": {
      "base_num_feature": 32,
      "num_conv_per_stage": 3,
      "conv_kernel": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
      "pool_kernel": [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
      "enable_ema": true
    },
    "decoder_architecture_setup": {
      "totalseg": [32, 3, [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
      "hn": [32, 3, [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    },
    "decoders": {
      "totalseg": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        101, 102, 103],
      "hn": [
        142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
        179, 180, 181, 182, 183]
    },
    "weights_for_decoders":{
      "totalseg": 1,
      "chest": 1,
      "hn": 3
    },
    "model_training_setup": {
      "decoders": {
        "all": [0.3, 0.8, 1e-3, 5000]
      }
    }
  }
}
```

### Training decoding heads 
Please refer to the following example to prepare the training/testing JSON file. 
```json
{
  "clnet_enable_ddp": false,
  "clnet_general_encoder": "Decoder_StructSeg_OAR22",
  "GeneralEncoder": {
    "task": "Task001_SuperGeneralEncoder",
    "fold": "all",
    "no_mirroring": true,
    "continue_training": false,
    "load_only_encoder": true,
    "pretrain_model_name": "final"
  },

  "Decoder_StructSeg_OAR22": {
    "train_order": 0,
    "task": "Task016_StructSeg_OAR22",
    "pretrain_model_name": "latest",
    "continue_training": false,
    "decoder_update": true,
    "finetune_encoder": false,
    "fold": 0,
    "no_mirroring": true,
    "optimizer": "adamw",
    "amsgrad": true,
    "warmup": 40,
    "plot_network": false,
    "decoders": {
      "BrainStem": 1,
      "Eyes": [2, 3],
      "Lens": [4, 5],
      "OpticNerve": [6, 7],
      "Chiasm": 8,
      "TempLobe": [9, 10],
      "Pituitary": 11,
      "Parotid": [12, 13],
      "InnerEar": [14, 15],
      "MidEar": [16, 17],
      "TMJ": [18, 19],
      "SpinalCord": 20,
      "Mandible": [21, 22]
    },
    "supporting": {
      "Lens": ["Eyes"],
      "OpticNerve": ["Eyes"],
      "Chiasm": ["Eyes", "OpticNerve"]
    },
    "decoder_architecture_setup": {
      "BrainStem": {
        "base_num_feature": 8, 
        "num_conv_per_stage": 2, 
        "conv_kernel": [[1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
        "enable_ema": true
      },
      "Eye_L": [16, 2, [3, 3, 3], [1, 3, 3], [3, 3, 3], [1, 3, 3], [3, 3, 3], true],
      "Eye_R": [4, 2, [3, 3, 3], [1, 3, 3], [3, 3, 3], [1, 3, 3], [3, 3, 3]],
      "Lens_L": [1, 2, [3, 3, 3], [1, 3, 3], [3, 3, 3], [1, 3, 3], [3, 3, 3]],
      "Lens_R": [1, 2, [3, 3, 3], [1, 3, 3], [3, 3, 3], [1, 3, 3], [3, 3, 3]],
      "OpticNerve_L": [1, 2, [3, 3, 3], [1, 3, 3], [3, 3, 3], [1, 3, 3], [3, 3, 3]],
      "OpticNerve_R": [1, 2, [3, 3, 3], [1, 3, 3], [3, 3, 3], [1, 3, 3], [3, 3, 3]],
      "Chiasm": [1, 2, [3, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
      "TempLobe_L": [1, 2, [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
      "TempLobe_R": [1, 2, [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
      "Pituitary": [1, 2, [3, 3, 3], [1, 3, 3], [3, 3, 3], [1, 3, 3], [3, 3, 3]],
      "Parotid_L": [1, 2, [3, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
      "Parotid_R": [1, 2, [3, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
      "InnerEar_L": [1, 2, [3, 3, 3], [1, 3, 3], [3, 3, 3], [1, 3, 3], [3, 3, 3]],
      "InnerEar_R": [1, 2, [3, 3, 3], [1, 3, 3], [3, 3, 3], [1, 3, 3], [3, 3, 3]],
      "MidEar_L": [1, 2, [3, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [3, 3, 3]],
      "MidEar_R": [1, 2, [3, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [3, 3, 3]],
      "TMJ_L": [1, 2, [3, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [3, 3, 3]],
      "TMJ_R": [1, 2, [3, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [3, 3, 3]],
      "SpinalCord": [1, 2, [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
      "Mandible_L": [1, 2, [3, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
      "Mandible_R": [1, 2, [3, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]]
    },
    "model_training_setup": {
      "decoders": {
        "BrainStem": {
          "foreground_sampling_decay": [0.2, 0.5],
          "lr": 1e-3,
          "epoch": 500,
          "load_pretrain": true,
          "prune": true
        },
        "Eye_L": [0, 0.5, 1e-3, 500, true, true],
        "Eye_R": [0, 0.5, 1e-3, 500, true],
        "Lens_L": [0.8, 1, 5e-3, 500, true],
        "Lens_R": [0.8, 1, 5e-3, 500, true],
        "OpticNerve_L": [0.8, 1, 5e-3, 800],
        "OpticNerve_R": [0.8, 1, 5e-3, 800],
        "Chiasm": [0.8, 1, 1e-3, 500],
        "TempLobe_L": [0, 1, 1e-3, 500],
        "TempLobe_R": [0, 1, 1e-3, 500],
        "Pituitary": [0, 1, 1e-3, 500],
        "Parotid_L": [0, 1, 1e-3, 500],
        "Parotid_R": [0, 1, 1e-3, 500],
        "InnerEar_L": [0, 1, 1e-3, 500],
        "InnerEar_R": [0, 1, 1e-3, 500],
        "MidEar_L": [0, 1, 1e-3, 500],
        "MidEar_R": [0, 1, 1e-3, 500],
        "TMJ_L": [0, 1, 1e-3, 500],
        "TMJ_R": [0, 1, 1e-3, 500],
        "SpinalCord": [0, 1, 1e-3, 500],
        "Mandible_L": [0, 1, 1e-3, 500],
        "Mandible_R": [0, 1, 1e-3, 500]
      },
      "supporting": {
        "Lens_L": [0.5, 1, 1e-3, 200],
        "Lens_R": [0.5, 1, 1e-3, 200],
        "Chiasm": [0.5, 1, 1e-3, 200]
      },
      "patch_size": {
        "Lens_L": [32, 64, 64],
        "Lens_R": [32, 64, 64],
        "OpticNerve_L": [32, 64, 64],
        "OpticNerve_R": [32, 64, 64],
        "Chiasm": [32, 64, 64]
      },
      "batch_size": {
        "Lens_L": 24
      }
    }
  }
}
```

### Configuration Common Parameters
***
***Required Parameters***
- `clnet_general_encoder` Specify the task for loading the general encoder
- `Task: str` The name of the task
- `decoders: (int, list)` The label(s) of the decoding head

***
***Optional Single-value Parameters***
- `clnet_enable_ddp (false)`: Set to "true" to enable DDP training setup
- `clnet_network (3d_fullres)`: Specify the 3D/2D network
- `clnet_trainer (clNetTrainerV2_SelectiveChannelDA_Continual_Decoding_Ensemble)`: Specify network trainer
- `full_network (false`': Set to "true" to use a full UNet architecture to train general encoder
- `fold (0)`: The training fold, e.g., 0, 1, 2, 3, 'all'
- `pretrain_model_name (null)`: Specify the absolute model pth or model keywords, e.g., best, latest, final
- `continue_training: (false)`: Set to true to load previously trained model and perform continue training 
- `decoder_update: (true)`: Set to true to update duplicated decoders using only the EMA update (w.o. pruning). 
Set to false to re-prune the decoder using current training task's dataset. 
- `finetune_encoder (false)`: Set to true to fine-tune the general encoder. The fine-tuned general encoder is stored separately in for each training head.
- `no_mirroring (false)`: Set to "true" to ignore mirroring data augmentation
- `optimizer (AdamW)`: Specify the training optimizer, e.g., SGD, Adam, AdamW
- `amsgrad (false)`: Set to true to activate amsgrad in adam-related optimizer.
- `plot_network (true)`: Set to "false" to disable the network plot
***

***Optional Multiple-value parameters***
- `encoder_architecture_setup (null)`: Specify the encoder architecture. If the architecture setup of the encoder is not specified, 
clNet will use the architecture setup in "**configuration.py**".
  - [`base num of feature`, `num of conv per stage`, `upper kernel → bottom kernel`, `pool_kernel` `enable EMA`]
    - Default: {
      - base_num_feature: 32, 
      - num_conv_per_stage: 2, 
      - conv_kernel: [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
      - pool_kernel: [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
      - enable_ema: true}

- `decoder_architecture_setup (null)`: Specify the decoding head CNN architecture. If the architecture setup of the 
decoding head is null, clNet will try to search for potential pretrained models in the working directory. 
If the model is found, clNet will load weights, else train from scratch using default setting.
  - [`base num of feature`, `num of conv per stage`, `upper kernel → bottom kernel`, `enable EMA`]
    - Default (simple version): [32, 2, [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], true]
    - Default (detailed version): {
      - base_num_feature: recommend_base_num_features (32, 24, 16), 
      - num_conv_per_stage: 2, 
      - conv_kernel: [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
      - enable_ema: true}
- `model_training_setup`: Specify the training scheme for each decoding head. The training setup requires at least 
If decoding head is null or left empty, clNet will not perform training of the decoding head.
  - `patch_size` {dict ([64, 160, 160])}: Specify the input image patch size. 
  - `batch_size`{dict (2)}: Specify the batch size for each decoding head
  - `decoders` {dict}: Specify the training setup for each decoding head.
  - `supporting` {dict}: Specify the training setup for each supporting head.
    - [`foreground sampling lower rate`, `foreground sampling upper rate`, `LR, epochs`, `load pretrain model`, `prune`,`ema`]
    - Default (simple version): [0.6, 0.8，1e-3, 1000, False, True]
    - Default (detailed version): {
      - foreground_sampling_decay: [0.6, 0.8]，
      - lr: 1e-3, 
      - epoch: 1000, 
      - load_pretrain: False,
      - prune: True
      - ema: True}
- `supporting (null)`: Specify a list of the supporting organ(s)

***
***Automated Pruning & EMA Updating***
- Automated Pruning
  - The automated pruning is based on the "Lottery Ticket Pruning". By default, each decoding head is targeted for pruning.
  - The decoding head is initially trained for 955 epochs, and then iteratively pruned for at most 15 epochs using predefined pruning ratios.
  - The pruned decoding head is trained for additional 30 epochs to recovery possible performance loss.  
  - Pruning acts by removing `weight` from the parameters and replacing it with a new parameter called `weight_orig` (i.e. appending "_orig" to the initial parameter name). 
`weight_orig` stores the unpruned version of the tensor. The bias was not pruned, so it will remain intact.
  - The pruning mask generated by the pruning is saved as a module buffer named `weight_mask` (i.e. appending "_mask" to the initial parameter name).
  - Once done pruning, the pruned connections (i.e., 0s in `weight_mask`) are not updated in training.
  - (Attention) If the `prune` is set to `False`, the pruned connections are reset (i.e., all 1s in `weight_mask`), s.t., all `weight` from the parameters are target for updating regardless of previous pruned masks. 
- EMA Updating
  - By default, all modules are target for EMA updating. 
  - The EMA module is updated every training iteration, using an Alpha=0.9999 for encoder and Alpha=0.999 for decoding heads. 
  - The EMA module of the decoding head is updated only after the pruning is done.


***
***Hard Object Segmentation***

- `1st` Try to warm up the network when using the Adam or AdamW optimizer.
  - E.g., Apply warm up scheme ⮕ "warmup": 40,
- `2nd` Finetune the previously trained decoding head. Lower both the foreground sampling rate and learning rate. 
  - E.g., Lens ⮕ from `train from scratch` setup [0.8, 1, 1e-2, 500] to `finetuning` setup [0.3, 0.8, 1e-3, 500]
- `3rd` If possible, apply feature-level-supporting to the small object segmentation.
  - E.g., Using `Eyes` to support `Lens` segmentation ⮕ "supporting": {"Lens": ["Eyes"]}

The framework will then automatically adapt to better small object segmentation setups. 

***
***Inference***

Please use the training/testing JSON file for inference. 
