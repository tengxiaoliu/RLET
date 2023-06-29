import re
import os
import json
import math
import csv
import pickle
import numpy as np
from fastNLP import Metric, Callback, print
import torch
import transformers
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, AutoConfig, AutoModel


transformers.logging.set_verbosity_error()
utils_tokenizer = None
verbose = False


class EBRLMetric(Metric):
    def __init__(self, verbose=False, has_end=True):
        super().__init__()
        self.reward, self.hits, self.all_correct, self.policy_reward = 0, 0, 0, 0
        self.pred_steps, self.gold_steps, self.total_cases = 0, 0, 0
        self.len_1, self.len_2, self.len_3, self.len_4, self.len_5 = 0, 0, 0, 0, 0

        self.discount = 0.99
        self.verbose = verbose
        self.f1_sum = 0
        self.has_end = has_end  # EBRL metric with distractors, end token

        self.reward_per_node = 0

    def update(self, loss, pred, target, reward, all_sents):

        if 'end' in pred[-1]:
            pred = pred[:-1]
        if 'end' in target[-1]:
            target = target[:-1]

        print(f"pred: {pred}")
        print(f"target: {target}")

        hits = 0
        total_reward = 0
        all_correct = 1

        aggre_leaves = aggregate_ancestor(target)
        pred_leaves = {}

        pred_map_gold_inter = {}
        for i, pred_step in enumerate(pred):
            leaves = []
            aligned_pred_step = []
            for name in pred_step:
                if name[0] == 'i':
                    leaves.extend(pred_leaves[name])
                    aligned_pred_step.append(pred_map_gold_inter[name])
                else:
                    leaves.append(name)
                    aligned_pred_step.append(name)
            leaves = list(set(leaves))
            pred_leaves['int' + str(i + 1)] = leaves

            max_sim = 0
            map_gold = 0
            for j, gold_leaves in enumerate(aggre_leaves):
                jaccard_sim = jaccard_similarity(leaves, gold_leaves)
                if jaccard_sim > max_sim:
                    max_sim = jaccard_sim
                    map_gold = j
            if max_sim > 0:
                pred_map_gold_inter['int' + str(i + 1)] = 'int' + str(map_gold + 1)
            else:
                pred_map_gold_inter['int' + str(i + 1)] = 'int0'

            gold_step = target[map_gold]

            if set(gold_step) == set(aligned_pred_step):
                hits += 1
                total_reward += 1
            else:
                total_reward -= 1
                all_correct = 0

            if self.verbose:
                inter_name = 'int' + str(i + 1)
                print(f"   [{pred_step[0]}] {all_sents[pred_step[0]]}\n"
                      f"   [{pred_step[1]}] {all_sents[pred_step[1]]}\n"
                      f"----> [{inter_name}] {all_sents[inter_name].strip('.')};")

        if len(target) != len(pred):
            all_correct = 0
        if all_correct == 1:
            total_reward = 5
            if len(target) == 1:
                self.len_1 += 1
            elif len(target) == 2:
                self.len_2 += 1
            elif len(target) == 3:
                self.len_3 += 1
            elif len(target) == 4:
                self.len_4 += 1
            elif len(target) > 4:
                self.len_5 += 1

        p = hits / len(pred)
        r = hits / len(target)
        if p + r != 0:
            self.f1_sum += 2 * p * r / (p + r)
        else:
            self.f1_sum += 0

        self.reward += total_reward
        self.hits += hits
        self.all_correct += all_correct
        self.pred_steps += len(pred)
        self.gold_steps += len(target)
        self.total_cases += 1
        self.policy_reward += reward

        if self.verbose:
            print(f"Case {self.total_cases}, step_ac {all_correct}, "
                  f"step_f1 {round(2 * p * r / (p + r), 6) if p + r != 0 else 0}.\n"
                  f"Step F1: {round(self.f1_sum / self.total_cases, 6)}, "
                  f"Step AC: {round(self.all_correct / self.total_cases, 6)}\n"
                  f"================\n\n")

    def get_metric(self, reset=True):

        outputs = {'F1': round(self.f1_sum / self.total_cases, 6),
                   'AllCorrect': round(self.all_correct / self.total_cases, 6),
                   'Reward': round(self.reward / self.total_cases, 6),
                   'PolicyReward': self.policy_reward / self.total_cases, 'CaseNum': self.total_cases,
                   'AC_1': self.len_1, 'AC_2': self.len_2, 'AC_3': self.len_3, 'AC_4': self.len_4, 'AC_5': self.len_5
                   }

        if reset:
            self.reward, self.hits, self.all_correct, self.policy_reward = 0, 0, 0, 0
            self.pred_steps, self.gold_steps, self.total_cases = 0, 0, 0
            self.len_1, self.len_2, self.len_3, self.len_4, self.len_5 = 0, 0, 0, 0, 0
            self.f1_sum = 0
        return outputs


class EBPredMetric(Metric):
    def __init__(self, test=False, out_path=None, pred_para=False, verbose=False):
        super().__init__()
        self.reward, self.hits, self.all_correct, self.policy_reward = 0, 0, 0, 0
        self.pred_steps, self.gold_steps, self.total_cases = 0, 0, 0
        self.len_1, self.len_2, self.len_3, self.len_4, self.len_5 = 0, 0, 0, 0, 0

        self.discount = 0.99
        self.test = test
        self.verbose = verbose

        self.pred_lst = []
        self.out_path = out_path
        self.pred_para = pred_para
        self.f1_sum = 0

    def update(self, loss, pred, target, all_sents, reward):

        if 'end' in pred[-1]:
            pred = pred[:-1]
        if 'end' in target[-1]:
            target = target[:-1]

        str_pred = self.to_EB_pred(pred, all_sents)
        self.pred_lst.append(str_pred)

        hits = 0
        total_reward = 0
        all_correct = 1

        aggre_leaves = aggregate_ancestor(target)
        pred_leaves = {}

        pred_map_gold_inter = {}
        for i, pred_step in enumerate(pred):
            leaves = []
            aligned_pred_step = []
            for name in pred_step:
                if name[0] == 'i':
                    leaves.extend(pred_leaves[name])
                    aligned_pred_step.append(pred_map_gold_inter[name])
                else:
                    leaves.append(name)
                    aligned_pred_step.append(name)
            leaves = list(set(leaves))
            pred_leaves['int' + str(i + 1)] = leaves

            max_sim = 0
            map_gold = 0
            for j, gold_leaves in enumerate(aggre_leaves):
                jaccard_sim = jaccard_similarity(leaves, gold_leaves)
                if jaccard_sim > max_sim:
                    max_sim = jaccard_sim
                    map_gold = j
            if max_sim > 0:
                pred_map_gold_inter['int' + str(i + 1)] = 'int' + str(map_gold + 1)
            else:
                pred_map_gold_inter['int' + str(i + 1)] = 'int0'

            gold_step = target[map_gold]

            if set(gold_step) == set(aligned_pred_step):
                hits += 1
                total_reward += 1
            else:

                total_reward -= 1
                all_correct = 0

        if len(target) != len(pred):
            all_correct = 0
        if all_correct == 1:
            if len(target) == 1:
                self.len_1 += 1
            elif len(target) == 2:
                self.len_2 += 1
            elif len(target) == 3:
                self.len_3 += 1
            elif len(target) == 4:
                self.len_4 += 1
            elif len(target) > 4:
                self.len_5 += 1

        p = hits / len(pred)
        r = hits / len(target)
        if p + r != 0:
            self.f1_sum += 2 * p * r / (p + r)
        else:
            self.f1_sum += 0
        self.reward += total_reward
        self.hits += hits
        self.all_correct += all_correct
        self.pred_steps += len(pred)
        self.gold_steps += len(target)
        self.total_cases += 1
        self.policy_reward += (-loss)

        if self.verbose:
            print(f"Case {self.total_cases}, step_ac {all_correct}, "
                  f"step_f1 {round(2 * p * r / (p + r), 6) if p + r != 0 else 0}.\n"
                  f"Step F1: {round(self.f1_sum / self.total_cases, 6)}, "
                  f"Step AC: {round(self.all_correct / self.total_cases, 6)}\n"
                  f"================\n\n")

    def to_EB_pred(self, pred, all_sents):
        # change pred to EB format for evaluation
        ret = "$proof$ ="
        for i, p in enumerate(pred):
            if p[0] == p[1]:
                tmp_str = ' ' + str(p[0]) + ' -> '
            else:
                tmp_str = ' ' + str(p[0]) + ' & ' + str(p[1]) + ' -> '
            if i < len(pred) - 1:
                inter_name = 'int' + str(i + 1)
                tmp_str += inter_name + ': ' + all_sents[inter_name].strip('.') + ';'
            else:
                inter_name = 'hypothesis'
                tmp_str += inter_name + ';'
            ret += tmp_str

            if self.verbose:
                print(f"   [{p[0]}] {all_sents[p[0]]}\n"
                      f"   [{p[1]}] {all_sents[p[1]]}\n"
                      f"----> [{inter_name}] {all_sents[inter_name].strip('.')};")

        return ret

    def get_metric(self, reset=True):
        prec = self.hits / self.pred_steps
        rec = self.hits / self.gold_steps

        outputs = {'F1': round(self.f1_sum / self.total_cases, 6),
                   'AllCorrect': round(self.all_correct / self.total_cases, 6),
                   'Reward': round(self.reward / self.total_cases, 6),
                   'PolicyReward': self.policy_reward / self.total_cases, 'CaseNum': self.total_cases,
                   'AC_1': self.len_1, 'AC_2': self.len_2, 'AC_3': self.len_3, 'AC_4': self.len_4, 'AC_5': self.len_5
                   }

        # write tsv file
        if self.out_path is not None:
            with open(os.path.join(os.getcwd(), self.out_path), 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                for record in self.pred_lst:
                    writer.writerow([record])
            print("[Metric] Saved EB format predictions as", self.out_path)
        else:
            print("[Metric] Save direction is None.")

        if reset:
            self.reward, self.hits, self.all_correct, self.policy_reward = 0, 0, 0, 0
            self.pred_steps, self.gold_steps, self.total_cases = 0, 0, 0
            self.len_1, self.len_2, self.len_3, self.len_4, self.len_5 = 0, 0, 0, 0, 0
            self.f1_sum = 0
            self.pred_lst = []

        return outputs


class EBPredMetricWithPara(Metric):
    def __init__(self, test=False, out_path=None, has_end=False, pred_para=True, verbose=True, inter_cache_path=None):
        super().__init__()
        self.reward, self.hits, self.all_correct, self.policy_reward = 0, 0, 0, 0
        self.pred_steps, self.gold_steps, self.total_cases = 0, 0, 0
        self.len_1, self.len_2, self.len_3, self.len_4, self.len_5 = 0, 0, 0, 0, 0
        self.discount = 0.99

        self.test = test

        self.pred_lst = []
        self.out_path = out_path
        self.has_end = has_end  # EBRL metric with distractors, end token
        self.pred_para = pred_para
        self.para = ParaPattern(inter_cache_path=inter_cache_path)
        print("[PredParaMetric] Use ParaPattern intermediate statement generation.")

        self.verbose = verbose

        self.f1_sum = 0

    def update(self, loss, pred, target, all_sents, reward):

        if 'end' in pred[-1]:
            pred = pred[:-1]
        if 'end' in target[-1]:
            target = target[:-1]

        str_pred = self.to_EB_pred(pred, all_sents)
        self.pred_lst.append(str_pred)

        hits = 0
        total_reward = 0
        all_correct = 1

        aggre_leaves = aggregate_ancestor(target)
        pred_leaves = {}

        pred_map_gold_inter = {}
        for i, pred_step in enumerate(pred):
            leaves = []
            aligned_pred_step = []
            for name in pred_step:
                if name[0] == 'i':
                    leaves.extend(pred_leaves[name])
                    aligned_pred_step.append(pred_map_gold_inter[name])
                else:
                    leaves.append(name)
                    aligned_pred_step.append(name)
            leaves = list(set(leaves))
            pred_leaves['int' + str(i + 1)] = leaves

            max_sim = 0
            map_gold = 0
            for j, gold_leaves in enumerate(aggre_leaves):
                jaccard_sim = jaccard_similarity(leaves, gold_leaves)
                if jaccard_sim > max_sim:
                    max_sim = jaccard_sim
                    map_gold = j
            if max_sim > 0:
                pred_map_gold_inter['int' + str(i + 1)] = 'int' + str(map_gold + 1)
            else:
                pred_map_gold_inter['int' + str(i + 1)] = 'int0'

            gold_step = target[map_gold]

            if set(gold_step) == set(aligned_pred_step):
                hits += 1
                total_reward += 1 * (self.discount ** i)
            else:
                total_reward -= 1 * (self.discount ** i)
                all_correct = 0

        if len(target) != len(pred):
            all_correct = 0
        if all_correct == 1:
            if len(target) == 1:
                self.len_1 += 1
            elif len(target) == 2:
                self.len_2 += 1
            elif len(target) == 3:
                self.len_3 += 1
            elif len(target) == 4:
                self.len_4 += 1
            elif len(target) > 4:
                self.len_5 += 1

        p = hits / len(pred)
        r = hits / len(target)
        if p + r != 0:
            self.f1_sum += 2 * p * r / (p + r)
        else:
            self.f1_sum += 0

        self.reward += total_reward
        self.hits += hits
        self.all_correct += all_correct
        self.pred_steps += len(pred)
        self.gold_steps += len(target)
        self.total_cases += 1
        self.policy_reward += reward

        if self.verbose:
            print(f"Case {self.total_cases}, step_ac {all_correct}, "
                  f"step_f1 {round(2 * p * r / (p + r), 6) if p + r != 0 else 0}.\n"
                  f"Step F1: {round(self.f1_sum / self.total_cases, 6)}, "
                  f"Step AC: {round(self.all_correct / self.total_cases, 6)}\n"
                  f"================\n\n")

    def to_EB_pred(self, pred, all_sents):
        # change pred to EB format for evaluation
        ret = "$proof$ ="
        eb_pred = pred
        for i, p in enumerate(eb_pred):
            if p[0] == p[1]:
                tmp_str = ' ' + str(p[0]) + ' -> '
            else:
                tmp_str = ' ' + str(p[0]) + ' & ' + str(p[1]) + ' -> '
            if i < len(eb_pred) - 1:
                inter_name = 'int' + str(i + 1)
                new_inter_text = self.get_inter(all_sents[p[0]], all_sents[p[1]])
                all_sents[inter_name] = new_inter_text
                tmp_str += inter_name + ': ' + new_inter_text.strip('.') + ';'
            else:
                inter_name = 'hypothesis'
                tmp_str += inter_name + ';'
            ret += tmp_str

            if self.verbose:
                print(f"   [{p[0]}] {all_sents[p[0]]}\n"
                      f"   [{p[1]}] {all_sents[p[1]]}\n"
                      f"----> [{inter_name}] {all_sents[inter_name].strip('.')};")
        return ret

    def get_inter(self, sent1, sent2):
        new_inter_text = self.para.get_intermediate_text(sent1, sent2)
        return new_inter_text

    def get_metric(self, reset=True):
        prec = self.hits / self.pred_steps
        rec = self.hits / self.gold_steps

        outputs = {'F1': round(self.f1_sum / self.total_cases, 6),
                   'AllCorrect': round(self.all_correct / self.total_cases, 6),
                   'Reward': round(self.reward / self.total_cases, 6),
                   'PolicyReward': self.policy_reward / self.total_cases, 'CaseNum': self.total_cases,
                   'AC_1': self.len_1, 'AC_2': self.len_2, 'AC_3': self.len_3, 'AC_4': self.len_4, 'AC_5': self.len_5
                   }

        # write tsv file
        if self.out_path is not None:
            with open(os.path.join(os.getcwd(), self.out_path), 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                for record in self.pred_lst:
                    writer.writerow([record])
            print("[Metric] Saved EB format predictions as", self.out_path)
        else:
            print("[Metric] Save direction is None.")

        if reset:
            self.reward, self.hits, self.all_correct, self.policy_reward = 0, 0, 0, 0
            self.pred_steps, self.gold_steps, self.total_cases = 0, 0, 0
            self.len_1, self.len_2, self.len_3, self.len_4, self.len_5 = 0, 0, 0, 0, 0
            self.f1_sum = 0

            self.pred_lst = []

        self.para.save_inter_cache()

        return outputs


class Task3EBPredMetric(Metric):
    def __init__(self, test=False, out_path=None, has_end=True, verbose=True, pred_para=False, inter_cache_path=None):
        super().__init__()
        self.reward, self.hits, self.all_correct, self.policy_reward = 0, 0, 0, 0
        self.pred_steps, self.gold_steps, self.total_cases = 0, 0, 0
        self.len_1, self.len_2, self.len_3, self.len_4, self.len_5 = 0, 0, 0, 0, 0
        self.discount = 0.99

        self.test = test
        self.verbose = verbose

        self.pred_lst = []
        self.out_path = out_path
        self.has_end = has_end  # EBRL metric with distractors, end token

        self.para = None
        if pred_para:
            self.para = ParaPattern(inter_cache_path=inter_cache_path)
            print("[Task3PredParaMetric] Use ParaPattern intermediate statement generation.")


    def update(self, loss, pred, target, all_sents):

        if 'end' in pred[-1]:
            pred = pred[:-1]
        str_pred = self.to_EB_pred(pred, all_sents)
        self.pred_lst.append(str_pred)

        self.pred_steps += len(pred)
        self.total_cases += 1
        self.policy_reward += (-loss.item())

    def to_EB_pred(self, pred, all_sents):
        # change pred to EB format for evaluation
        ret = "$proof$ ="
        eb_pred = pred
        for i, p in enumerate(eb_pred):
            if p[0] == p[1]:
                tmp_str = ' ' + str(p[0]) + ' -> '
            else:
                tmp_str = ' ' + str(p[0]) + ' & ' + str(p[1]) + ' -> '
            if i < len(eb_pred) - 1:
                inter_name = 'int' + str(i+1)
                if self.para is not None:
                    new_inter_text = self.get_inter(all_sents[p[0]], all_sents[p[1]])
                    all_sents[inter_name] = new_inter_text
                else:
                    new_inter_text = all_sents[inter_name]
                tmp_str += inter_name + ': ' + new_inter_text.strip('.') + ';'
            else:
                inter_name = 'hypothesis'
                tmp_str += inter_name + ';'
            ret += tmp_str

            if self.verbose:
                print(f"   [{p[0]}] {all_sents[p[0]]}\n"
                      f"   [{p[1]}] {all_sents[p[1]]}\n"
                      f"----> [{inter_name}] {all_sents[inter_name].strip('.')};")
        return ret

    def get_metric(self, reset=True):

        outputs = {'Reward': self.reward / self.total_cases,
                   'PolicyReward': self.policy_reward / self.total_cases,
                   'CaseNum': self.total_cases
                   }

        # write tsv file
        if self.out_path is not None:
            with open(os.path.join(os.getcwd(), self.out_path), 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                for record in self.pred_lst:
                    writer.writerow([record])
            print("[Metric] Saved EB format predictions as", self.out_path)
        else:
            print("[Metric] Save direction is None.")

        if reset:
            self.reward, self.hits, self.all_correct, self.policy_reward = 0, 0, 0, 0
            self.pred_steps, self.gold_steps, self.total_cases = 0, 0, 0
            self.len_1, self.len_2, self.len_3, self.len_4, self.len_5 = 0, 0, 0, 0, 0

            self.pred_lst = []

        return outputs

    def get_inter(self, sent1, sent2):
        new_inter_text = self.para.get_intermediate_text(sent1, sent2)
        if new_inter_text.strip(".") == sent1.strip(".").strip(' ') \
                or new_inter_text.strip(".") == sent2.strip(".").strip(' '):
            new_inter_text = get_intermediate_sentence(sent1, sent2)
        return new_inter_text

class SaveLastModelCallBack(Callback):
    def __init__(self):
        super().__init__()

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if is_better_eval:
            model_name = '_'.join(['best_ckp_model', self.trainer.start_time])
            self.trainer._save_model(self.model, model_name, only_param=True)


class tSNEMetric(Metric):
    def __init__(self, test=True, out_path=None):
        super().__init__()
        self.feat = None
        self.sent = []
        self.out_path = out_path

    def update(self, feat, sent):
        if feat is None and sent is None:
            return

        feat = feat.cpu().detach().numpy()
        sent = sent.cpu().detach().tolist()

        if self.feat is None:
            self.feat = feat
        else:
            self.feat = np.append(self.feat, feat, axis=0)

        self.sent.extend(sent)

    def get_metric(self, reset=True):
        with open(os.path.join(self.out_path, 'feat_gold.npy'), 'wb') as f:
            np.save(f, self.feat)

        with open(os.path.join(self.out_path, "sent_ids_gold.pkl"), "wb") as fp:  # Pickling
            pickle.dump(self.sent, fp)

        ret = {'sent': len(self.sent), 'feat': self.feat.shape}

        if reset:
            self.feat = None
            self.sent = []

        return ret


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / max(1, union)


def aggregate_ancestor(gold):
    int2leaf = {}
    aggre_leaves = []
    for idx, step in enumerate(gold):
        leaves = []
        inter_name = 'int' + str(idx + 1)

        for name in step:
            if name[0] == 'i':
                leaves.extend(int2leaf[name][1])
            else:
                leaves.append(name)

        int2leaf[inter_name] = [list(step), list(set(leaves))]
        aggre_leaves.append(leaves)
    return aggre_leaves


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


class ScheduledSamplingCallback(Callback):
    def __init__(self, mode='linear'):
        super().__init__()
        self.mode = mode
        self.linear_lb = 0.75  # linear lower bound
        self.exp_k = 0.9  # exp k, 0 < k < 1
        self.sigmoid_k = 2  # k > 1

        self.eps = 1.0

    def on_train_epoch_end(self, trainer):
        if self.mode == 'linear':
            self.eps = self.eps - (1 - self.linear_lb) / trainer.n_epochs
        elif self.mode == 'exp':
            self.eps *= self.exp_k
        else:
            self.eps = self.sigmoid_k / (self.sigmoid_k + math.exp(trainer.epoch / self.sigmoid_k))

        trainer.model.policy.set_epsilon(self.eps)


def get_intermediate_sentence(sent1, sent2):
    return sent1.strip('.').strip(' ') + ', and ' + sent2.strip('.').strip(' ') + '.'


def set_tokenizer(args):
    global utils_tokenizer
    utils_tokenizer = AutoTokenizer.from_pretrained(args.policy_model_config, local_files_only=True)


def get_tokenize_inputs(text, args, hypo):
    global utils_tokenizer
    if utils_tokenizer is None:
        set_tokenizer(args)
    hypo_text = [[hypo[0].strip('.') + '. ', t[0].strip().strip('.') + '. ' + t[1].strip().strip('.') + '.'] for t in
                 text]

    inputs = utils_tokenizer(text=hypo_text,
                             truncation=True,
                             padding='longest',
                             max_length=args.max_seq_length,
                             return_tensors='pt')

    device = int(args.device[0])
    if 'roberta' in utils_tokenizer.name_or_path:
        return {'input_ids': inputs.input_ids.to(device),
                'attention_mask': inputs.attention_mask.to(device)}
    else:
        return {'input_ids': inputs.input_ids.to(device),
                'attention_mask': inputs.attention_mask.to(device),
                'token_type_ids': inputs.token_type_ids.to(device)
                }


class ParaPattern(object):
    def __init__(self, device='cuda:1', inter_cache_path=None):

        self.model_path = 'outputs/examples/para/ckpt'
        self.capital = True
        print("[WARNING] using full parapattern model!")

        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model.config.update({
            'early_stopping': True,
            'max_length': 128,
            'min_length': 8,
            'prefix': '',
            'decoder_start_token_id': self.tokenizer.bos_token_id
        })
        if 't5' in self.model_path:
            self.model.config.update({
                'prefix': 'combine: ',
                'decoder_start_token_id': self.tokenizer.pad_token_id
            })
            print("==============Updated T5 config")

        self.device = device
        self.model.to(self.device)
        print("[ParaPattern] model loaded at", self.model_path)

        self.cache_dict = {}

        self.inter_cache_path = inter_cache_path

        if self.inter_cache_path is not None:
            self.load_inter_cache(self.inter_cache_path)

        self.model.eval()

    def get_intermediate_text(self, sent1, sent2):

        s_lst = [sent1, sent2]
        s_lst.sort()
        input_cache_key = '+'.join(s_lst)

        if input_cache_key in self.cache_dict.keys():
            return self.cache_dict[input_cache_key]

        if self.capital:
            sent1 = capitalize(sent1)
            sent2 = capitalize(sent2)

        input_text = sent1.strip('.') + '. ' + sent2.strip('.') + '.'
        batch = self.tokenizer(input_text, padding='longest', return_tensors='pt')

        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        generated = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        conclusion = decoded[0].replace('"', '').replace('\'', '').strip()
        conclusion = normalize(conclusion)
        self.cache_dict[input_cache_key] = conclusion

        return conclusion

    def save_inter_cache(self):
        if self.inter_cache_path is None:
            self.inter_cache_path = os.path.join(self.model_path, 'para_cache_t2.json')

        json.dump(self.cache_dict, open(self.inter_cache_path, 'w'))
        print(f"[Policy-deduction] Saved inter cache at {self.inter_cache_path}.")

    def load_inter_cache(self, inter_cache_path):
        self.cache_dict = json.load(open(inter_cache_path, 'r'))
        print(f"[Policy-deduction] Loaded inter cache at {inter_cache_path}.")


def has_overlap(sent1, sent2):
    stopwords = {'a', 'the', 'is', 'are'}
    s1 = set(sent1.strip('.').split()) - stopwords
    s1 = {re.sub(r"^(\w{3,}?)(?:es|s|ing|e|ed)$", r'\1', word) for word in s1}
    s2 = set(sent2.strip('.').split()) - stopwords
    s2 = {re.sub(r"^(\w{3,}?)(?:es|s|ing|e|ed)$", r'\1', word) for word in s2}
    return not s1.isdisjoint(s2)


def chunk(it, n):
    c = []
    for x in it:
        c.append(x)
        if len(c) == n:
            yield c
            c = []
    if len(c) > 0:
        yield c

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

def to_binary_path(path):
    """
    Transform a gold path (only premises/source) to a list of binary steps.
    :param path: gold path
    :return: a list of binary steps
    """

    binary_path = []
    inter_idx = 1
    ori2new = {}  # projection from original inter index to new inter index

    for step_idx, step in enumerate(path):
        tgt = 'int' + str(step_idx + 1)

        if len(step) == 1:
            if step[0][0] == 'i':
                one_name = ori2new[step[0]]
            else:
                one_name = step[0]
            ori2new[tgt] = one_name

        elif len(step) == 2:
            one_step = []
            for one in step:
                if one[0] == 'i':
                    one_step.append(ori2new[one])
                else:
                    one_step.append(one)
            binary_path.append(one_step)
            new_tgt = 'int' + str(inter_idx)
            ori2new[tgt] = new_tgt
            inter_idx += 1

        elif len(step) > 2:
            one_1 = step[0]
            if one_1[0] == 'i':
                one_1 = ori2new[one_1]
            for i in range(len(step) - 1):
                one_2 = step[i + 1]
                if one_2[0] == 'i':
                    one_2 = ori2new[one_2]
                binary_path.append([one_1, one_2])

                new_tgt = 'int' + str(inter_idx)
                if i == len(step) - 2:
                    ori2new[tgt] = new_tgt
                one_1 = new_tgt
                inter_idx += 1
    return binary_path

def have_overlap(sent1, sent2):
    stopwords = {'a', 'the', 'is', 'are', 'of', '.', 'kind', 'to', '/', 'on'}
    s1 = set(sent1.strip('.').split()) - stopwords
    s1 = {re.sub(r"^(\w{3,}?)(?:es|s|ing|e|ed|ies|y|ion|ions)$", r'\1', word) for word in s1}
    s2 = set(sent2.strip('.').split()) - stopwords
    s2 = {re.sub(r"^(\w{3,}?)(?:es|s|ing|e|ed|ies|y|ion|ions)$", r'\1', word) for word in s2}
    return not s1.isdisjoint(s2)