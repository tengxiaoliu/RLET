# Load full EntailmentBank data and EvalEB data
# from v11, but can be used as unified dataloader

from typing import Dict, Union
from fastNLP import Instance, DataSet
from fastNLP.io import Loader, DataBundle, Pipe
from fastNLP import print

import copy
import random
import json
import jsonlines
import itertools
from utils import have_overlap, normalize

MAX_STEP_NUM = 7
DIST_NUM = 5
ONLY_BINARY = False

class OldBinaryEBLoader(Loader):
    r"""
    transform each multi-way tree into binary tree
    """

    def __init__(self):
        super().__init__()

    def _load(self, path: str = None):

        ds = DataSet()
        cnt = 0

        with open(path, 'r', encoding='utf-8') as f:
            for sample in jsonlines.Reader(f):

                multi_sent = False
                evidences = copy.deepcopy(sample['meta']['triples'])
                ori_inters = sample['meta']['intermediate_conclusions']
                new_allsents = copy.deepcopy(evidences)
                hypo = sample['hypothesis']

                step_src = []
                step_tgt = []
                proof = sample['proof'].split("; ")[:-1]

                inter_idx = 1
                inters = {}
                ori2new = {}  # projection from original inter index to new inter index

                gold_evidence = []
                ori_distractors = sample['meta']['distractors']
                distractors = []

                # filter distractors: keep sentences that have overlap with hypothesis
                for k in ori_distractors:
                    if have_overlap(new_allsents[k], hypo):
                        distractors.append(k)

                if len(distractors) < DIST_NUM:
                    distractors = ori_distractors

                # iterate each step, construct a sentence pair selection data for each step
                for step_idx, step in enumerate(proof):
                    src, tgt = step.split(':')[0].split(' -> ')
                    src_lst = src.split(' & ')

                    for src in src_lst:
                        if src[0] == 's':
                            gold_evidence.append(src)

                    if tgt == 'hypothesis':
                        tgt = sample['meta']['hypothesis_id']

                    if len(src_lst) == 1:
                        if src_lst[0][0] == 'i':
                            one_name = ori2new[src_lst[0]]
                        else:
                            one_name = src_lst[0]
                        ori2new[tgt] = one_name

                        multi_sent = True

                    elif len(src_lst) == 2:
                        one_step = []
                        for one in src_lst:
                            if one[0] == 'i':
                                one_step.append(ori2new[one])
                            else:
                                one_step.append(one)
                        step_src.append(one_step)
                        new_tgt = 'int' + str(inter_idx)
                        ori2new[tgt] = new_tgt
                        new_allsents[new_tgt] = ori_inters[tgt]
                        inter_idx += 1

                    elif len(src_lst) > 2:
                        multi_sent = True

                        one_1 = src_lst[0]
                        if one_1[0] == 'i':
                            one_1 = ori2new[one_1]
                        # change to multi binary steps
                        for i in range(len(src_lst) - 1):
                            one_2 = src_lst[i + 1]
                            if one_2[0] == 'i':
                                one_2 = ori2new[one_2]
                            step_src.append([one_1, one_2])

                            new_tgt = 'int' + str(inter_idx)
                            if i == len(src_lst) - 2:
                                ori2new[tgt] = new_tgt
                                new_allsents[new_tgt] = ori_inters[tgt]
                            else:
                                new_allsents[new_tgt] = normalize(new_allsents[one_1]).strip('.') + ', and ' + \
                                                        normalize(new_allsents[one_2]).strip('.')
                            one_1 = new_tgt
                            inter_idx += 1

                    if tgt == 'hypothesis':
                        tgt = sample['meta']['hypothesis_id']
                    step_tgt.append(tgt)

                # if the length of the chain exceeds MAX_STEP_NUM, truncate the chain
                if len(step_src) >= MAX_STEP_NUM:
                    cnt += 1

                    step_src = step_src[:MAX_STEP_NUM - 1]
                    truncated_all_sent_name = list(itertools.chain.from_iterable(step_src))
                    # evidences = {}
                    gold_evidence = []
                    for k in new_allsents.keys():
                        if k[0] == 'i' and int(k[-1]) < MAX_STEP_NUM:
                            inters[k] = new_allsents[k]
                        elif k[0] == 's' and k in set(truncated_all_sent_name):
                            gold_evidence.append(k)
                else:
                    for k in new_allsents.keys():
                        if k[0] == 'i':
                            inters[k] = new_allsents[k]

                # build initial candidates list
                candidates = []
                cand_names = []
                labels = []
                if len(step_src) == 0:
                    step_src = [['sent1', 'sent1']]
                    print("[DataLoader] Encounter only one step with one sentence.")
                chosen = step_src[0]

                random.shuffle(distractors)
                gold_dist_evidences = gold_evidence + distractors[:DIST_NUM]
                gold_dist_evi_keys = set(gold_dist_evidences)
                for key1 in set(gold_dist_evidences):
                    gold_dist_evi_keys.remove(key1)  # for task 2, remove if len(evi_keys) = 0
                    for key2 in gold_dist_evi_keys:
                        if key1 in chosen and key2 in chosen:
                            label = 1
                        else:
                            label = 0

                        if have_overlap(evidences[key1], evidences[key2]):
                            candidates.append([normalize(hypo),
                                               normalize(evidences[key1]) + ' ' + normalize(evidences[key2])])
                            cand_names.append([key1, key2])
                            labels.append(label)

                gold_dist_dict = {k: evidences[k] for k in list(set(gold_dist_evidences))}

                if ONLY_BINARY and multi_sent:
                    continue

                if len(candidates) == 0:
                    print(evidences)
                ds.append(Instance(evidence=gold_dist_dict,
                                   intermediate=inters,
                                   candidate=candidates,
                                   candidate_name=cand_names,
                                   label=labels,
                                   step_source=step_src,
                                   hypothesis=hypo
                                   ))
        print("[DataLoader] Truncated %d long chains to length %d." % (cnt, MAX_STEP_NUM))
        return ds

    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        r"""
        :return: 返回的 :class:`~fastNLP.io.DataBundle`
        """
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class BinaryEBLoader(Loader):
    r"""
    from ver8, transform each multi-way tree into binary tree
    """

    def __init__(self):
        super().__init__()

    def _load(self, path: str = None):

        ds = DataSet()
        cnt = 0

        with open(path, 'r', encoding='utf-8') as f:
            for one_sample in jsonlines.Reader(f):
                sample = copy.deepcopy(one_sample)
                sample = truncate_sample(sample)

                multi_sent = False
                evidences = copy.deepcopy(sample['meta']['triples'])
                ori_inters = sample['meta']['intermediate_conclusions']
                new_allsents = copy.deepcopy(evidences)
                hypo = sample['hypothesis']

                step_src = []
                step_tgt = []
                proof = sample['proof'].split("; ")[:-1]

                inter_idx = 1
                inters = {}
                ori2new = {}  # projection from original inter index to new inter index

                gold_evidence = []
                ori_distractors = sample['meta']['distractors']
                distractors = []

                # filter distractors: keep sentences that have overlap with hypothesis
                for k in ori_distractors:
                    if have_overlap(new_allsents[k], hypo):
                        distractors.append(k)

                if len(distractors) < DIST_NUM:
                    distractors = ori_distractors

                # iterate each step, construct a sentence pair selection data for each step
                for step_idx, step in enumerate(proof):
                    src, tgt = step.split(':')[0].split(' -> ')
                    src_lst = src.split(' & ')

                    for src in src_lst:
                        if src[0] == 's':
                            gold_evidence.append(src)

                    if tgt == 'hypothesis':
                        tgt = sample['meta']['hypothesis_id']

                    if len(src_lst) == 1:
                        if src_lst[0][0] == 'i':
                            one_name = ori2new[src_lst[0]]
                        else:
                            one_name = src_lst[0]
                        ori2new[tgt] = one_name

                        multi_sent = True

                    elif len(src_lst) == 2:
                        one_step = []
                        for one in src_lst:
                            if one[0] == 'i':
                                one_step.append(ori2new[one])
                            else:
                                one_step.append(one)
                        step_src.append(one_step)
                        new_tgt = 'int' + str(inter_idx)
                        ori2new[tgt] = new_tgt
                        new_allsents[new_tgt] = ori_inters[tgt]
                        inter_idx += 1

                    elif len(src_lst) > 2:
                        multi_sent = True

                        one_1 = src_lst[0]
                        if one_1[0] == 'i':
                            one_1 = ori2new[one_1]
                        # change to multi binary steps
                        for i in range(len(src_lst) - 1):
                            one_2 = src_lst[i + 1]
                            if one_2[0] == 'i':
                                one_2 = ori2new[one_2]
                            step_src.append([one_1, one_2])

                            new_tgt = 'int' + str(inter_idx)
                            if i == len(src_lst) - 2:
                                ori2new[tgt] = new_tgt
                                new_allsents[new_tgt] = ori_inters[tgt]
                            else:
                                new_allsents[new_tgt] = normalize(new_allsents[one_1]).strip('.') + ', and ' + \
                                                        normalize(new_allsents[one_2]).strip('.')
                            one_1 = new_tgt
                            inter_idx += 1

                    if tgt == 'hypothesis':
                        tgt = sample['meta']['hypothesis_id']
                    step_tgt.append(tgt)

                # if the length of the chain exceeds MAX_STEP_NUM, truncate the chain
                if len(step_src) >= MAX_STEP_NUM:
                    cnt += 1

                    step_src = step_src[:MAX_STEP_NUM - 1]
                    truncated_all_sent_name = list(itertools.chain.from_iterable(step_src))
                    # evidences = {}
                    gold_evidence = []
                    for k in new_allsents.keys():
                        if k[0] == 'i' and int(k[-1]) < MAX_STEP_NUM:
                            inters[k] = new_allsents[k]
                        elif k[0] == 's' and k in set(truncated_all_sent_name):
                            gold_evidence.append(k)
                else:
                    for k in new_allsents.keys():
                        if k[0] == 'i':
                            inters[k] = new_allsents[k]

                # build initial candidates list
                candidates = []
                cand_names = []
                labels = []
                if len(step_src) == 0:
                    step_src = [['sent1', 'sent1']]
                    print("[DataLoader] Encounter only one step with one sentence.")
                chosen = step_src[0]

                random.shuffle(distractors)
                gold_dist_evidences = gold_evidence + distractors[:DIST_NUM]
                gold_dist_evi_keys = set(gold_dist_evidences)
                for key1 in set(gold_dist_evidences):
                    gold_dist_evi_keys.remove(key1)  # for task 2, remove if len(evi_keys) = 0
                    for key2 in gold_dist_evi_keys:
                        if key1 in chosen and key2 in chosen:
                            label = 1
                        else:
                            label = 0

                        if have_overlap(evidences[key1], evidences[key2]):
                            candidates.append([normalize(hypo),
                                               normalize(evidences[key1]) + ' ' + normalize(evidences[key2])])
                            cand_names.append([key1, key2])
                            labels.append(label)

                gold_dist_dict = {k: evidences[k] for k in list(set(gold_dist_evidences))}

                if ONLY_BINARY and multi_sent:
                    continue

                if len(candidates) == 0:
                    print(evidences)
                ds.append(Instance(evidence=gold_dist_dict,
                                   intermediate=inters,
                                   candidate=candidates,
                                   candidate_name=cand_names,
                                   label=labels,
                                   step_source=step_src,
                                   hypothesis=hypo
                                   ))
        print("[DataLoader] Truncated %d long chains to length %d." % (cnt, MAX_STEP_NUM))
        return ds

    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        r"""
        :return: 返回的 :class:`~fastNLP.io.DataBundle`
        """
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class EvalEBLoader(Loader):
    r"""
    DataLoader for EB style evaluation, keep every multi-way step
    """

    def __init__(self):
        super().__init__()

    def _load(self, path: str = None):
        # short version for training
        is_test = True if 'test' in path else False

        ds = DataSet()
        num_failed = 0
        num_single_cand = 0
        with open(path, 'r', encoding='utf-8') as f:

            if 'filter' in path:
                jsonsamples = json.load(f)
            else:
                jsonsamples = jsonlines.Reader(f)

            for sample in jsonsamples:

                evidences = sample['meta']['triples']
                inters = sample['meta']['intermediate_conclusions']
                hypo = sample['hypothesis']

                step_src = []
                step_tgt = []
                proof = sample['proof'].split("; ")[:-1]

                # iterate each step, construct a sentence pair selection data for each step
                for step_idx, step in enumerate(proof):
                    src, tgt = step.split(':')[0].split(' -> ')
                    src_lst = src.split(' & ')

                    if len(src_lst) == 1:
                        one = src_lst[0]
                        src_lst = [one, one]

                    step_src.append(src_lst)
                    if tgt == 'hypothesis':
                        tgt = sample['meta']['hypothesis_id']
                    step_tgt.append(tgt)

                # build initial candidates list
                candidates = []
                cand_names = []
                labels = []
                chosen = step_src[0]
                keys_set = set(evidences.keys())
                for key1 in evidences.keys():
                    keys_set.remove(key1)
                    for key2 in keys_set:
                        if key1 in chosen and key2 in chosen:
                            label = 1
                        else:
                            label = 0

                        candidates.append([normalize(hypo),
                                           normalize(evidences[key1]) + ' ' + normalize(evidences[key2])])
                        cand_names.append([key1, key2])
                        labels.append(label)

                if len(candidates) == 0:
                    print("Added one step one candidate case.")
                    candidates.append([normalize(hypo),
                                       normalize(evidences['sent1']) + ' ' + normalize(evidences['sent1'])])
                    cand_names.append(['sent1', 'sent1'])
                    labels.append(1)

                ds.append(Instance(evidence=evidences,
                                   intermediate=inters,
                                   candidate=candidates,
                                   candidate_name=cand_names,
                                   label=labels,
                                   step_source=step_src,
                                   step_target=step_tgt,
                                   hypothesis=hypo
                                   ))
        print("Excluded " + str(num_failed) + " cases with only one step source.")
        print("Total single step cand:", num_single_cand)
        return ds

    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        r"""
        :return: 返回的 :class:`~fastNLP.io.DataBundle`
        """
        datasets = {name: self._load(path) for name, path in paths.items()}
        print(datasets['test'][1]['evidence'])
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class EBDataPipe(Pipe):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = args.max_seq_length
        self.n_proc = 8

    def process(self, data_bundle: DataBundle) -> DataBundle:
        def tokenize(ins):
            inputs = self.tokenizer(text=ins['candidate'],
                                    truncation=True,
                                    max_length=self.max_length)
            if 'roberta' in self.tokenizer.name_or_path:
                return {'input_ids': inputs.input_ids,
                        'attention_mask': inputs.attention_mask
                        }
            else:
                return {'input_ids': inputs.input_ids,
                        'attention_mask': inputs.attention_mask,
                        'token_type_ids': inputs.token_type_ids
                        }

        data_bundle.apply_more(tokenize, num_proc=self.n_proc, progress_desc='tokenize')
        data_bundle.delete_field('candidate')
        ignore_fields = ['evidence', 'intermediate', 'candidate_name', 'step_source', 'step_target', 'hypothesis']
        for field in ignore_fields:
            data_bundle.set_pad(field, pad_val=None)

        return data_bundle

    def process_from_file(self, paths: Union[str, Dict[str, str]], test=False, task3=False) -> DataBundle:
        if task3:
            data_bundle = Task3EvalEBLoader().load(paths)
            print("[Pipe] Loading data for task3.")
        elif test:
            data_bundle = EvalEBLoader().load(paths)
            print("[Pipe] Loading data for evaluation.")
        else:
            data_bundle = BinaryEBLoader().load(paths)
            print("[Pipe] Loading data for training.")
        return self.process(data_bundle)


class Task3EvalEBLoader(Loader):
    r"""
    DataLoader for EB style evaluation, keep every multi-way step
    This is specifically for task 3, where proof is not available.
    """

    def __init__(self):
        super().__init__()

    def _load(self, path: str = None):

        ds = DataSet()
        num_failed = 0
        num_single_cand = 0
        with open(path, 'r', encoding='utf-8') as f:
            if 'filter' in path:
                jsonsamples = json.load(f)
            else:
                jsonsamples = jsonlines.Reader(f)

            for sample in jsonsamples:

                evidences = sample['meta']['triples']
                hypo = sample['hypothesis']

                # build initial candidates list
                candidates = []
                cand_names = []
                keys_set = set(evidences.keys())
                for key1 in evidences.keys():
                    keys_set.remove(key1)
                    for key2 in keys_set:
                        candidates.append([normalize(hypo),
                                           normalize(evidences[key1]) + ' ' + normalize(evidences[key2])])
                        cand_names.append([key1, key2])

                # to fit the unified dataset format, save these fields
                inters, labels, step_src, step_tgt = [0], [0], [0], [0]
                ds.append(Instance(evidence=evidences,
                                   intermediate=inters,
                                   candidate=candidates,
                                   candidate_name=cand_names,
                                   label=labels,
                                   step_source=step_src,
                                   step_target=step_tgt,
                                   hypothesis=hypo
                                   ))
        print("Excluded " + str(num_failed) + " cases with only one step source.")
        print("[Task3 Eval data] Processed")
        return ds

    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        r"""
        :return: 返回的 :class:`~fastNLP.io.DataBundle`
        """
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class Task3EBDataPipe(Pipe):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = args.max_seq_length
        self.n_proc = 8

    def process(self, data_bundle: DataBundle) -> DataBundle:
        def tokenize(ins):
            inputs = self.tokenizer(text=ins['candidate'],
                                    truncation=True,
                                    max_length=self.max_length)
            if 'roberta' in self.tokenizer.name_or_path:
                return {'input_ids': inputs.input_ids,
                        'attention_mask': inputs.attention_mask
                        }
            else:
                return {'input_ids': inputs.input_ids,
                        'attention_mask': inputs.attention_mask,
                        'token_type_ids': inputs.token_type_ids
                        }

        data_bundle.apply_more(tokenize, num_proc=self.n_proc, progress_desc='tokenize')
        data_bundle.delete_field('candidate')
        ignore_fields = ['evidence', 'intermediate', 'candidate_name', 'step_source', 'step_target', 'hypothesis']
        for field in ignore_fields:
            data_bundle.set_pad(field, pad_val=None)

        return data_bundle

    def process_from_file(self, paths: Union[str, Dict[str, str]], test=False) -> DataBundle:
        data_bundle = Task3EvalEBLoader().load(paths)
        print("[Task3 Pipe] Loading data for evaluation.")
        return self.process(data_bundle)


def truncate_sample(sample):
    # truncate samples from backward

    proof = sample['proof'].split("; ")[:-1]
    if len(proof) <= MAX_STEP_NUM:
        return sample

    ori_facts = copy.deepcopy(sample['meta']['triples'])
    ori_inters = sample['meta']['intermediate_conclusions']
    ori_distractors = sample['meta']['distractors']
    ori_allsents = copy.deepcopy(ori_facts)
    ori_allsents.update(ori_inters)
    hypo = sample['hypothesis']

    fact_ptr = 26
    inter_ptr = 1

    ori2new = {}
    new_facts = {}
    new_inters = {}
    new_proofs = []

    for step_idx, step in enumerate(proof[-MAX_STEP_NUM:]):

        src, tgt = step.split(':')[0].split(' -> ')
        if tgt == 'hypothesis':
            tgt = sample['meta']['hypothesis_id']
        src_lst = list(set(src.split(' & ')))  # remove duplicated sents
        new_src_lst = []

        for s in src_lst:

            new_s = s
            if s in ori2new.keys():
                new_s = ori2new[s]
            elif s[0] == "i":
                new_s = f"sent{fact_ptr}"
                ori2new[s] = new_s
                fact_ptr += 1
                new_facts[new_s] = ori_allsents[s]

            new_src_lst.append(new_s)

        new_inter_name = f"int{inter_ptr}"
        inter_ptr += 1
        ori2new[tgt] = new_inter_name
        new_inters[new_inter_name] = ori_allsents[tgt]

        new_pr = f"{' & '.join(new_src_lst)} -> {new_inter_name}: {new_inters[new_inter_name]}; "
        new_proofs.append(new_pr)

    # take care of distractors

    new_sample = copy.deepcopy(sample)
    ori_facts.update(new_facts)

    new_sample['proof'] = ''.join(new_proofs)
    new_sample['meta']['triples'] = ori_facts
    new_sample['meta']['intermediate_conclusions'] = new_inters
    new_sample['meta']['hypothesis_id'] = f"int{inter_ptr - 1}"

    return new_sample