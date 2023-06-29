import os
import time
import torch
from torch.optim import AdamW
from fastNLP import TorchWarmupCallback, FitlogCallback, CheckpointCallback, LoadBestModelCallback, TorchDataLoader
from fastNLP import RandomSampler, SequentialSampler, prepare_torch_dataloader
from fastNLP import cache_results, Trainer, Evaluator, Event, print

from transformers import AutoTokenizer, AutoModel

from parse import parse_args
from pipe import EBDataPipe
from model import EBRLModel
from utils import EBRLMetric, ScheduledSamplingCallback, EBPredMetric, EBPredMetricWithPara

import logging
import fitlog


def do_train():
    # 1. logs and args
    logging.basicConfig(level=logging.INFO)
    logging.info('Start logging ଘ(੭ˊᵕˋ)੭')
    args = parse_args()
    # fitlog.debug()
    fitlog.commit(__file__)
    fitlog.set_log_dir('logs')
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)
    if args.seed != 0:
        fitlog.set_rng_seed(args.seed)
        print("Set random seed as", args.seed)
    else:
        fitlog.set_rng_seed()

    save_dir = args.model_type
    if args.learning_rate != 1e-5:
        save_dir += '_lr' + str(args.learning_rate)
    if args.dropout != 0.1:
        save_dir += '_drop' + str(args.dropout)
    if args.K != 2:
        save_dir += '_K' + str(args.K)
    if args.gradient_accumulation_steps != 1:
        save_dir += '_acc' + str(args.gradient_accumulation_steps)
    if args.discount != 0.99:
        save_dir += '_dis' + str(args.discount)
    if args.warmup_proportion != 0:
        save_dir += '_w' + str(args.warmup_proportion) + '_' + str(args.schedule)[0]
    if args.seed != 42:
        save_dir += '_rnd' + str(args.seed)
    save_dir += '_ep' + str(args.num_epochs)
    save_dir = os.path.join('outputs', args.model_save_dir, save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    paths = {'train': os.path.join(args.data_dir, 'train.jsonl'),
             'dev': os.path.join(args.data_dir, 'dev.jsonl'),
             'test': os.path.join(args.data_dir, 'test.jsonl')}

    # 2. data
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model_config, local_files_only=True)
    data_save_path = os.path.join(args.data_dir, args.data_save_path)
    print("data_save_path:", data_save_path)

    @cache_results(data_save_path, _refresh=args.refresh_data)
    def get_data(test=False):
        pipe = EBDataPipe(tokenizer, args)
        _data_bundle = pipe.process_from_file(paths, test)
        return _data_bundle

    train_data_bundle = get_data(test=False)
    # print(train_data_bundle)
    train_data = train_data_bundle.get_dataset('train')

    eval_data_bundle = get_data(_cache_fp=os.path.join(args.data_dir, args.test_data), test=True)
    logging.info("Using [Eval] data to test.")
    dev_data = eval_data_bundle.get_dataset('dev')
    test_data = eval_data_bundle.get_dataset('test')

    train_dl = prepare_torch_dataloader(train_data, batch_size=args.train_batch_size, sampler=RandomSampler(train_data))
    dev_dl = prepare_torch_dataloader(dev_data, batch_size=args.eval_batch_size, sampler=SequentialSampler(dev_data))
    test_dl = prepare_torch_dataloader(test_data, batch_size=args.eval_batch_size, sampler=SequentialSampler(test_data))

    # 3. model
    model = EBRLModel(args)

    if len(args.device) == 1:
        device = int(args.device[0])
    else:
        device = [int(d) for d in args.device]

    if args.do_train:
        if args.pred_para:
            pred_save_dir = os.path.join(save_dir, 'pred_para.tsv')
        else:
            pred_save_dir = os.path.join(save_dir, 'pred.tsv')

        if args.policy_model_path is not None:
            warmup_model_dict = torch.load(args.policy_model_path)
            model_dict = model.state_dict()

            # 1) filter out unnecessary keys
            recover_dict = {k: v for k, v in warmup_model_dict.items() if k in model_dict}
            # 2) overwrite entries in the existing state dict
            model_dict.update(recover_dict)
            # 3) load the new state dict
            model.load_state_dict(model_dict)
            logging.info('Loaded parameters from warming up model at ' + str(args.policy_model_path) + '.')
        else:
            pretrained_model = AutoModel.from_pretrained(args.policy_model_config)
            model = EBRLModel(args, ptm=pretrained_model)
            print("Use pretrained model parameter without warmup training.")

        # 4. loss, optimizer, metric
        # loss = LossInForward()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        metric = {"rlet": EBRLMetric()}

        # 5. callbacks and trainer

        model_save_dir = os.path.join(save_dir, "checkpoint")
        callbacks = [TorchWarmupCallback(warmup=args.warmup_proportion, schedule=args.schedule),
                     FitlogCallback(log_loss_every=200),
                     LoadBestModelCallback(save_folder=model_save_dir, only_state_dict=True),
                     CheckpointCallback(folder=model_save_dir, topk=2, monitor='F1#rlet#test',
                                        only_state_dict=True),

                     ScheduledSamplingCallback(args.sampling)]
        logging.info('Start training ଘ( ˊ•̥▵•)੭₎₎')

        # result_save_dir = os.path.join(save_dir, args.result_dir)

        result_save_dir = os.path.join(save_dir, args.result_dir)
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir, exist_ok=True)

        @Trainer.on(Event.on_before_backward(every=400))
        def print_loss(trainer, outputs):
            print(f"train_loss: {outputs['loss']}")

        if args.evaluate_before_train:
            t_begin = time.time()
            evaluator = Evaluator(model, driver='torch', device=device, dataloaders=test_dl,
                                  metrics=metric,
                                  progress_bar='rich',
                                  kwargs={"use_dist_sampler": False},
                                  fp16=args.fp16
                                  )

            evaluator.run()
            t_end = time.time()
            print(f'Finished evaluating on TEST {len(test_data)} examples.')
            print(f'Test (before train) duration: {t_end - t_begin} sec.')

        t_train_begin = time.time()

        trainer = Trainer(model=model,
                          train_dataloader=train_dl,
                          optimizers=optimizer,
                          driver='torch',
                          device=device,
                          evaluate_dataloaders={"dev": dev_dl, "test": test_dl},
                          metrics=metric,
                          callbacks=callbacks,
                          output_mapping=None,
                          n_epochs=args.num_epochs,
                          evaluate_every=args.eval_step,
                          accumulation_steps=args.gradient_accumulation_steps,
                          fp16=args.fp16,
                          torch_kwargs={'ddp_kwargs': {'find_unused_parameters': True}},
                          monitor='F1#rlet#test',
                          # overfit_batches=300
                          )

        trainer.run()
        t_train_end = time.time()
        logging.info(f'Training duration: {round((t_train_end - t_train_begin)/60, 4)} minutes, '
                     f'i.e., {round((t_train_end - t_train_begin)/3600, 4)} hours.')
    else:
        # model.load_state_dict(torch.load(args.model_path).state_dict())
        model.load_state_dict(torch.load(args.policy_model_path))
        logging.info('[Eval] Loaded actor parameters from test model at ' + str(args.policy_model_path) + '.')

        pred_save_dir = os.path.split(args.policy_model_path)[0]
        
        if args.pred_para:
            pred_save_dir = os.path.join(pred_save_dir, 'pred_para.tsv')
        else:
            pred_save_dir = os.path.join(pred_save_dir, 'pred.tsv')
        logging.info('[Eval] Will save pred at ' + str(pred_save_dir) + '.')

    logging.info('Start testing ଘ(੭*ˊᵕˋ)੭* ੈ♡‧₊˚')

    if args.pred_para:
        if args.do_train:
            verbose = False
        else:
            verbose = True
        metrics = {"rlet": EBPredMetricWithPara(test=True, out_path=pred_save_dir, verbose=verbose,
                                                 inter_cache_path=args.inter_cache_path, pred_para=args.pred_para)}
    else:
        metrics = {"rlet": EBPredMetric(test=True, out_path=pred_save_dir, verbose=False)}

    print(f"Set k_test = {args.K_test} in evaluation.")

    t_begin = time.time()
    evaluator = Evaluator(model, driver='torch', device=device, dataloaders=test_dl,
                          metrics=metrics,
                          progress_bar='rich',
                          kwargs={"use_dist_sampler": False},
                          fp16=args.fp16
                          )

    evaluator.run()
    t_end = time.time()
    print(f'Finished evaluating on TEST {len(test_data)} examples.')
    print(f'Test duration: {t_end - t_begin} sec.')

    fitlog.finish()


if __name__ == "__main__":
    do_train()
