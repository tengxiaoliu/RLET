import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # names and paths
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument("--data_save_path", default='full_data.pkl', type=str)
    parser.add_argument('--model_path', type=str, default=None,
                        help="The pretrained model to recover for RL, should be a path")
    parser.add_argument("--policy_model_config", default='microsoft/deberta-v3-base', type=str)
    parser.add_argument("--model_save_dir", default='checkpoint_task23', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--result_dir", default='result', type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--policy_model_path", default=None, type=str)
    parser.add_argument("--refresh_data", action="store_true")
    parser.add_argument("--inter_cache_path", default=None, type=str)
    parser.add_argument("--task3", action="store_true")
    parser.add_argument("--irgr", action="store_true")

    # training hyper parameters

    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--schedule", default='linear', type=str)
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup proportion over the training process.")
    parser.add_argument("--eval_step", default=500, type=int, help="Step interval for evaluation.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")

    # model parameters
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument('--K', default=2, type=int, help="beam size in beam search")
    parser.add_argument('--K_test', default=1, type=int, help="beam size in beam search during inference")
    parser.add_argument("--discount", default=0.99, type=float, help="discount weight when measuring reward")
    parser.add_argument("--path_max_length", default=20, type=int)
    parser.add_argument("--sampling", default='linear', type=str)
    parser.add_argument("--bleu_reward", action="store_true")
    parser.add_argument("--wrong_reward", default=-1, type=float, help="reward assigned to wrong node")
    parser.add_argument("--inter_mode", default='naive', type=str, help="choose from naive, para, hybrid")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test_data", default='data_hypo_strict_test.pkl', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--pred_para", action="store_true")
    parser.add_argument("--bleurt", action="store_true")
    parser.add_argument("--step_scorer", action="store_true")
    parser.add_argument("--evaluate_before_train", action="store_true")
    parser.add_argument("--filter", action="store_true")

    # DDP settings
    parser.add_argument("--device", default=0, type=list, help="GPU device")

    args = parser.parse_args()
    return args
