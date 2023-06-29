#!/bin/sh
export CUDA_VISIBLE_DEVICES=2

model_config=microsoft/deberta-v3-base
model_type=rl_t2
data_save_path=rl_t2.pkl
test_data=rl_t2_eval.pkl
inter_mode=naive
lr=1e-5
n_epochs=20
warmup=0.1
schedule=linear
train_bs=1
gradient_accumulation_steps=1
seed=900
# should end up with fastnlp_model.pkl.tar
policy_model_path=PATH_TO_WARMUP_MODEL_TASK23

python src/rl_task23/filter_sentences.py --filter task2 &&

python -u src/rl_task23/train.py \
    --data_dir data/entailment_trees_emnlp2021_data_v3/dataset/task_2/ \
    --data_save_path $data_save_path \
    --test_data $test_data \
    --policy_model_config $model_config \
    --model_type $model_type \
    --num_epochs $n_epochs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --warmup_proportion $warmup \
    --schedule $schedule \
    --train_batch_size $train_bs \
    --eval_batch_size 1 \
    --learning_rate $lr \
    --inter_mode $inter_mode \
    --eval_step -1 \
    --discount 0.99 \
    --K 2 \
    --K_test 1 \
    --device 0 \
    --seed $seed \
    --dropout 0.3 \
    --do_train \
    --filter \
    --policy_model_path $policy_model_path