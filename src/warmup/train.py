
import os

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core import Evaluator

from fastNLP.core import TorchWarmupCallback, CheckpointCallback, LoadBestModelCallback, \
    EarlyStopCallback, Event, FitlogCallback
# from fastNLP import TorchWarmupCallback, LoadBestModelCallback, CheckpointCallback, EarlyStopCallback

from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP.core.samplers import RandomSampler, SequentialSampler
from fastNLP import TorchDataLoader, print

from parse import parse_args
from pipe import EBPipe
from model import SentencePairModel
from utils import SaveLastModelCallBack, SentencePairMetric

import logging
import fitlog


def do_train():
    # 1. logs and args
    logging.basicConfig(level=logging.INFO)
    print('Start logging ଘ(੭ˊᵕˋ)੭')
    args = parse_args()
    # fitlog.debug()
    fitlog.commit(__file__)
    fitlog.set_log_dir('logs')
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)
    if args.seed != 0:
        fitlog.set_rng_seed(args.seed)
        print("Set random seed as", str(args.seed))
    else:
        fitlog.set_rng_seed()

    ptm = args.pretrained_path
    if "deberta" in args.pretrained_path:
        ptm = args.pretrained_path.split("/")[-1]
    save_name = '_'.join(['model', ptm, str(args.model_type)])
    if args.gradient_accumulation_steps != 1:
        save_name += '_acc' + str(args.gradient_accumulation_steps)
    save_name += '_lr' + str(args.learning_rate)
    if args.warmup_proportion != 0:
        save_name += '_w' + str(args.warmup_proportion)
    save_name += '_ep' + str(args.num_epochs)

    model_save_folder = os.path.join('outputs', args.model_save_dir, save_name)

    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder, exist_ok=True)

    paths = {'train': os.path.join(args.data_dir, 'train.jsonl'),
             'dev': os.path.join(args.data_dir, 'dev.jsonl'),
             'test': os.path.join(args.data_dir, 'test.jsonl')}

    # 2. data
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path, local_files_only=False)
    data_save_path = os.path.join(args.data_dir, args.data_save_path)

    @cache_results(data_save_path, _refresh=args.refresh_data)
    def get_data(test=False):
        pipe = EBPipe(tokenizer, args)
        _data_bundle = pipe.process_from_file(paths, test)
        return _data_bundle

    data_bundle = get_data()
    print(data_bundle)
    train_data = data_bundle.get_dataset('train')

    eval_data_bundle = get_data(_cache_fp=os.path.join(args.data_dir, args.test_data), test=True)
    logging.info("Using [Eval] data to test.")
    dev_data = eval_data_bundle.get_dataset('dev')
    test_data = eval_data_bundle.get_dataset('test')

    train_dl = TorchDataLoader(train_data, batch_size=args.train_batch_size, shuffle=False,
                               sampler=RandomSampler(train_data), num_workers=0
                               )
    dev_dl = TorchDataLoader(dev_data, batch_size=args.eval_batch_size, shuffle=False,
                             sampler=SequentialSampler(dev_data), num_workers=0
                             )
    test_dl = TorchDataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                              sampler=SequentialSampler(test_data), num_workers=0
                              )

    # 3. model
    pretrained_model = AutoModel.from_pretrained(args.pretrained_path, local_files_only=False)
    model = SentencePairModel(pretrained_model=pretrained_model, args=args)

    # 4. loss, optimizer, metric

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    metrics = {"SPMetric": SentencePairMetric()}

    # 5. callbacks and trainer
    callbacks = [LoadBestModelCallback(save_folder=model_save_folder, only_state_dict=True),
                 EarlyStopCallback(patience=8),
                 CheckpointCallback(folder=model_save_folder, topk=1, monitor='acc#SPMetric#dev',
                                    only_state_dict=True),
                 TorchWarmupCallback(warmup=args.warmup_proportion, schedule=args.schedule),
                 FitlogCallback(log_loss_every=1000)
                 ]
    print('Start training ଘ( ˊ•̥▵•)੭₎₎')

    if len(args.device) == 1:
        device = int(args.device[0])
    else:
        device = [int(d) for d in args.device]

    @Trainer.on(Event.on_before_backward(every=1000))
    def print_loss(trainer, outputs):
        print(f"train_loss: {outputs['loss']}")

    trainer = Trainer(model=model,
                      train_dataloader=train_dl,
                      optimizers=optimizer,
                      driver='torch',
                      device=device,
                      evaluate_dataloaders={"dev": dev_dl},
                      metrics=metrics,
                      callbacks=callbacks,
                      output_mapping=None,
                      n_epochs=args.num_epochs,
                      evaluate_every=-1,
                      accumulation_steps=args.gradient_accumulation_steps,
                      fp16=args.fp16,
                      torch_kwargs={'ddp_kwargs': {'find_unused_parameters': True}},
                      monitor='acc#SPMetric#dev'
                      )

    trainer.run()

    print('Start testing ଘ(੭*ˊᵕˋ)੭* ੈ♡‧₊˚')

    evaluator = Evaluator(model, driver='torch', device=device, dataloaders=test_dl,
                          metrics=metrics,
                          progress_bar='rich',
                          kwargs={"use_dist_sampler": False}
                          )
    evaluator.run()

    print('Finished evaluating on TEST with ' + str(len(test_data)) + ' examples.')

    fitlog.finish()


if __name__ == "__main__":
    do_train()
