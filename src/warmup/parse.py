import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # names and paths
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--data_save_path", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--pretrained_path", default='bert-base-uncased', type=str)
    parser.add_argument("--model_save_dir", default='checkpoint_warm', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--result_dir", default='result', type=str)
    parser.add_argument("--model_type", default='rlet', type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--load_model_path", type=str)

    parser.add_argument("--train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--schedule", default='constant', type=str)
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup proportion over the training process.")
    parser.add_argument("--eval_step", default=500, type=int, help="Step interval for evaluation.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--refresh_data", action="store_true")

    # model parameters
    parser.add_argument("--dropout", default=0.1, type=float)

    # DDP settings
    parser.add_argument("--device", default=0, type=list, help="GPU device")


    args = parser.parse_args()
    return args
