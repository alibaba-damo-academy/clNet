#!/bin/bash
# CUDA_LAUNCH_BLOCKING=1 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --master_port=12360 clnet/run/run_training_ddp.py clnet/training_cfg_json/CSS_GeneralEncoder.json
# torchrun --nproc_per_node=2  clnet/run/run_training_ddp.py clnet/training_cfg_json/CSS_GeneralEncoder.json
CUDA_VISIBLE_DEVICES='0' python clnet/run/run_training.py clnet/training_cfg_json/CSS_GeneralEncoder_ft.json

