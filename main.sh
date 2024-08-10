#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$pwd
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --prompt_data=AVA \
  --prompt_backbone=CLIP-RN50_no-pos \
  --prompt_ratio=1.0 \
  --target_data=AADB \
  --target_backbone=CLIP-RN50_no-pos \
  --batch_size=34 \
  --device=cuda \
  --seed=42