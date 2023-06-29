import re
import os
import json
from fastNLP import Metric, Callback
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.core.metrics.backend import Backend
from typing import Union

class SentencePairMetric(Metric):
    def __init__(self, backend: Union[str, Backend, None] = 'auto'):
        super().__init__()
        self.preds, self.labels, self.lengths = None, None, None

        self.register_element(name='acc_count', value=0, aggregate_method='sum', backend=backend)
        self.register_element(name='total', value=0, aggregate_method='sum', backend=backend)
        self.register_element(name='not_1_acc', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='not_1_total', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='acc_3', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='total_3', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='acc_4', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='total_4', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='acc_5', value=0, aggregate_method="sum", backend=backend)
        self.register_element(name='total_5', value=0, aggregate_method="sum", backend=backend)


    def update(self, target, pred, length):
        self.labels = target.view(-1, 1)
        self.preds = pred.view(-1, 1)
        self.lengths = length

        ptr = 0

        for len_idx, len_list in enumerate(self.lengths):
            for i, set_len in enumerate(len_list):
                if set_len == 0:
                    continue
                elif i + 1 < len(len_list) and len_list[i + 1] == 0:
                    ptr += set_len
                    continue
                elif (i + 1) % len(len_list) == 0 and self.labels[ptr: ptr + set_len].sum().item() <= 0:
                    ptr += set_len
                    continue

                assert self.labels[ptr: ptr + set_len].sum().item() == 1, \
                    str(ptr) + " - " + str(ptr + set_len) + " sum: " + str(self.labels[ptr: ptr + set_len].sum().item())

                pred_idx = torch.argmax(self.preds[ptr: ptr + set_len])
                label_idx = torch.argmax(self.labels[ptr: ptr + set_len])
                if pred_idx == label_idx:
                    self.acc_count += 1
                self.total += 1
                ptr += set_len

                if set_len > 1:
                    self.not_1_total += 1
                    if pred_idx == label_idx:
                        self.not_1_acc += 1
                    if set_len == 3:
                        self.total_3 += 1
                        if pred_idx == label_idx:
                            self.acc_3 += 1
                    elif set_len == 6:
                        self.total_4 += 1
                        if pred_idx == label_idx:
                            self.acc_4 += 1
                    elif set_len == 10:
                        self.total_5 += 1
                        if pred_idx == label_idx:
                            self.acc_5 += 1

    def get_metric(self, reset=True):
        # acc_strict: only count hit on set where number of candidates > 1
        evaluate_result = {'acc': round(self.acc_count.get_scalar() / (self.total.get_scalar() + 1e-12), 6),
                           'acc_strict': round(self.not_1_acc.get_scalar() / (self.not_1_total.get_scalar() + 1e-12), 6),
                           'acc_3': round(self.acc_3.get_scalar() / (self.total_3.get_scalar() + 1e-12), 6),
                           'acc_4': round(self.acc_4.get_scalar() / (self.total_4.get_scalar() + 1e-12), 6),
                           'acc_5': round(self.acc_5.get_scalar() / (self.total_5.get_scalar() + 1e-12), 6),
                           'total': self.total.get_scalar(),

                           }
        if reset:
            self.acc_count, self.total, self.not_1_acc, self.not_1_total = 0, 0, 0, 0
            self.acc_3, self.total_3, self.acc_4, self.total_4, self.acc_5, self.total_5 = 0, 0, 0, 0, 0, 0
        return evaluate_result


class SaveLastModelCallBack(Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self):
        model_name = '_'.join(['model', self.trainer.start_time, str(self.epoch)])
        self.trainer._save_model(self.model, model_name, only_param=True)



class PbarCallback(Callback):
    def __init__(self, pbar_every):
        super().__init__()
        self.pbar_every = pbar_every
        self._avg_loss = 0

    def on_backward_begin(self, loss):
        self._avg_loss += loss.item()
        if self.step % self.pbar_every == 0 or self.step == self.n_steps:
            _pbar_every = self.pbar_every if self.step % self.pbar_every == 0 else self.n_steps % self.pbar_every
            self.pbar.write("Training on train at Epoch {}/{}. Step:{}/{}: Loss: {:<6.5f}".format(
                self.epoch, self.n_epochs, self.step, self.n_steps, float(self._avg_loss) / _pbar_every))
            self._avg_loss = 0

def decapitalize(sent):
    if len(sent) > 1:
        return sent[0].lower() + sent[1:]
    else:
        return sent.lower()


def capitalize(sent):
    if len(sent) > 1:
        return sent[0].upper() + sent[1:]
    else:
        return sent.upper()


def normalize(sent):
    """
    add period to a sentence, and decapitalize
    """
    if sent.endswith('.'):
        return decapitalize(sent).strip()
    else:
        return decapitalize(sent).strip() + '.'
