#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1

model_config=microsoft/deberta-v3-base
model_type=rl_test
data_save_path=rl_t1.pkl
test_data=rl_t1_eval.pkl
inter_mode=naive
policy_model_path=outputs/examples/rl/fastnlp_model.pkl.tar
inter_cache_path=outputs/examples/para/para_cache_t2.json

python -u src/rl/train.py \
    --data_dir data/entailment_trees_emnlp2021_data_v3/dataset/task_1/ \
    --data_save_path $data_save_path \
    --test_data $test_data \
    --policy_model_config $model_config \
    --model_type $model_type \
    --eval_batch_size 1 \
    --inter_mode $inter_mode \
    --eval_step 500 \
    --discount 0.99 \
    --K 2 \
    --K_test 1 \
    --device 0 \
    --policy_model_path $policy_model_path \
    --pred_para \
    --inter_cache_path $inter_cache_path