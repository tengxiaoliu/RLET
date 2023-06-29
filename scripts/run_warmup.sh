#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1

python -u src/warmup/train.py \
    --device 01 \
    --data_dir data/warmup/task1 \
    --data_save_path warmup_train.pkl \
    --test_data warmup_eval.pkl \
    --num_epochs 20 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4 \
    --warmup_proportion 0.05 \
    --eval_step -1 \
    --model_type warmup_t1 \
    --pretrained_path microsoft/deberta-v3-base \
    --seed 42