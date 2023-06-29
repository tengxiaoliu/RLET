from typing import Dict, Union

from fastNLP import Instance, DataSet
from fastNLP.io import Loader, DataBundle, Pipe

import re
import json
import copy
import random

from utils import normalize

MAX_SET_LEN = 10
MAX_SENT_NUM = MAX_SET_LEN * (MAX_SET_LEN - 1) / 2 + 1  # max sent pair num in each instance


class EBLoader(Loader):
    r"""
    EntailmentBank sentence pair selection dataloader
    sentence list -> sentence pair with 0/1 label
    """

    def __init__(self):
        super().__init__()

    def _load(self, path: str = None):

        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            items = json.load(f)
            sent_pair_lst = []
            label_lst = []
            length_lst = []

            item_idx = 0
            split_flag = False
            while item_idx < len(items):
                if split_flag:
                    sentences = sent_dict2
                    split_flag = False
                    item_idx += 1
                else:
                    item = items[item_idx]
                    hypo = item['hypothesis']
                    if len(item['sentences']) > MAX_SET_LEN:
                        # if there are too many sentences, we'll split them as two sets
                        irrelevant = copy.deepcopy(item['sentences'])  # not chosen sentences
                        relevant = {}
                        for c in item['chosen']:
                            relevant[c] = irrelevant.pop(c)

                        sent_dict1 = {}
                        sent_dict2 = {}

                        for k_i, k in enumerate(irrelevant.keys()):
                            if k_i < MAX_SET_LEN - len(relevant):
                                # to avoid split set still exceeds MAX_SET_LEN
                                sent_dict1[k] = irrelevant[k]
                            elif MAX_SET_LEN - len(relevant) <= k_i < 2 * (MAX_SET_LEN - len(relevant)):
                                sent_dict2[k] = irrelevant[k]

                        sent_dict1.update(relevant)
                        sent_dict2.update(relevant)
                        assert len(sent_dict1) <= MAX_SET_LEN and len(sent_dict2) <= MAX_SET_LEN, "split not valid."

                        split_flag = True
                        sentences = sent_dict1
                    else:
                        split_flag = False
                        sentences = item['sentences']
                        item_idx += 1

                chosen = item['chosen']

                if len(label_lst) + len(sentences) * (len(sentences) - 1) / 2 + 1 > MAX_SENT_NUM:
                    assert 0 < len(label_lst) <= MAX_SENT_NUM, "data list is not valid: " + str(len(label_lst))
                    # padding
                    pad_num = 0
                    while len(label_lst) < MAX_SENT_NUM:
                        sent_pair_lst.append(['[PAD]', '[PAD]'])
                        pad_num += 1
                        label_lst.append(0)

                    length_lst.append(pad_num)
                    assert len(label_lst) == MAX_SENT_NUM and len(label_lst) == sum(length_lst) \
                           and len(label_lst) > 0 and len(length_lst) > 0, \
                        "length not consistent" + str(len(label_lst)) + " sum: " + str(sum(length_lst))
                    ds.append(Instance(sent_pairs=sent_pair_lst, length=length_lst, target=label_lst))
                    sent_pair_lst = []
                    label_lst = []
                    length_lst = []

                begin_num = len(label_lst)
                keys_set = set(sentences.keys())
                if len(chosen) == 2:
                    for key1 in sentences.keys():
                        keys_set.remove(key1)
                        for key2 in keys_set:
                            if key1 in chosen and key2 in chosen:
                                label = 1
                            else:
                                label = 0

                            if key1 == 'end':
                                sent_pair_lst.append([normalize(hypo), normalize(sentences[key2]) + ' ' +
                                                      normalize(sentences[key1])])
                            else:
                                sent_pair_lst.append([normalize(hypo), normalize(sentences[key1]) + ' ' +
                                                      normalize(sentences[key2])])
                            label_lst.append(label)
                else:
                    chosen_sents = [normalize(sentences[k]) for k in chosen]
                    sent_pair_lst.append([normalize(hypo), ' '.join(chosen_sents)])
                    label_lst.append(1)
                    # build negative samples
                    for key1 in sentences.keys():
                        keys_set.remove(key1)
                        for key2 in keys_set:
                            if key1 in chosen and key2 in chosen:
                                continue
                            else:
                                if key1 == 'end':
                                    sent_pair_lst.append([normalize(hypo), normalize(sentences[key2]) + ' ' +
                                                          normalize(sentences[key1])])
                                else:
                                    sent_pair_lst.append([normalize(hypo), normalize(sentences[key1]) + ' ' +
                                                          normalize(sentences[key2])])
                                label_lst.append(0)

                length_lst.append(len(label_lst) - begin_num)


            assert 0 < len(label_lst) <= MAX_SENT_NUM, "data list is null"
            # padding
            pad_num = 0
            while len(label_lst) < MAX_SENT_NUM:
                sent_pair_lst.append(['[PAD]', '[PAD]'])
                pad_num += 1
                label_lst.append(0)
            length_lst.append(pad_num)
            assert len(label_lst) == MAX_SENT_NUM and len(label_lst) == sum(length_lst) \
                   and len(label_lst) > 0 and len(length_lst) > 0, "length not consistent"
            ds.append(Instance(sent_pairs=sent_pair_lst, length=length_lst, target=label_lst))

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        datasets = {name: self._load(path) for name, path in paths.items()}
        print(datasets['train'][1]['sent_pairs'], datasets['train'][1]['length'], datasets['train'][1]['target'])
        print(datasets['dev'][1]['sent_pairs'], datasets['dev'][1]['length'], datasets['dev'][1]['target'])
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class EvalEBLoader(Loader):
    r"""
    EntailmentBank sentence pair selection dataloader
    sentence list -> sentence pair with 0/1 label
    """

    def __init__(self):
        super().__init__()

    def _load(self, path: str = None):

        ds = DataSet()
        excluded = 0
        with open(path, 'r', encoding='utf-8') as f:
            items = json.load(f)
            sent_pair_lst = []
            label_lst = []
            length_lst = []

            item_idx = 0

            while item_idx < len(items):

                item = items[item_idx]
                hypo = item['hypothesis']
                sentences = item['sentences']
                chosen = item['chosen']
                item_idx += 1

                begin_num = len(label_lst)
                keys_set = set(sentences.keys())

                if len(chosen) < 2:
                    excluded += 1
                    continue

                random.shuffle(chosen)
                chosen = chosen[:2]

                for key1 in sentences.keys():
                    keys_set.remove(key1)
                    for key2 in keys_set:
                        if key1 in chosen and key2 in chosen:
                            label = 1
                        else:
                            label = 0

                        if key1 == 'end':
                            sent_pair_lst.append([normalize(hypo), normalize(sentences[key2]) + ' ' +
                                                  normalize(sentences[key1])])
                        else:
                            sent_pair_lst.append([normalize(hypo), normalize(sentences[key1]) + ' ' +
                                                  normalize(sentences[key2])])
                        label_lst.append(label)

                length_lst.append(len(label_lst) - begin_num)
                assert len(label_lst) == sum(length_lst) and len(label_lst) > 0 and len(length_lst) > 0, \
                    "length not consistent"
                assert sum(label_lst) == 1, f"chosen: {chosen}"
                assert sum(length_lst) >= 1, f"sent_pair_lst: {sent_pair_lst}, sentences: {sentences}, " \
                                             f"length_lst: {length_lst}"
                ds.append(Instance(sent_pairs=sent_pair_lst, length=length_lst, target=label_lst))

                sent_pair_lst = []
                label_lst = []
                length_lst = []

        print(f"Excluded orphan sentences: {excluded}.")

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        datasets = {name: self._load(path) for name, path in paths.items()}
        print(datasets['dev'][0]['sent_pairs'], datasets['dev'][0]['length'], datasets['dev'][0]['target'])
        print(datasets['dev'][1]['sent_pairs'], datasets['dev'][1]['length'], datasets['dev'][1]['target'])
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class EBPipe(Pipe):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = args.max_seq_length
        self.n_proc = 8

    def process(self, data_bundle: DataBundle) -> DataBundle:
        # 对输入的DataBundle进行处理，然后返回该DataBundle
        def tokenize(ins):
            inputs = self.tokenizer(text=ins['sent_pairs'],
                                    truncation=True,
                                    max_length=self.max_length)
            if 'roberta' in self.tokenizer.name_or_path:
                return {'input_ids': inputs.input_ids,
                        'attention_mask': inputs.attention_mask}
            else:
                return {'input_ids': inputs.input_ids,
                        'attention_mask': inputs.attention_mask,
                        'token_type_ids': inputs.token_type_ids}

        data_bundle.apply_more(tokenize, num_proc=self.n_proc, progress_desc='tokenize')
        data_bundle.delete_field('sent_pairs')
        return data_bundle

    def process_from_file(self, paths: Union[str, Dict[str, str]], test=False) -> DataBundle:
        if test:
            data_bundle = EvalEBLoader().load(paths)
            print("[Pipe] Loading data for evaluation.")
        else:
            data_bundle = EBLoader().load(paths)
            print("[Pipe] Loading data for training.")
        return self.process(data_bundle)

