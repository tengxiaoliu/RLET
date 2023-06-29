import os
import sys
import copy
import numpy as np
import random
from fastNLP import print

import torch
from sacrebleu.metrics import BLEU
from utils import get_intermediate_sentence, ParaPattern, jaccard_similarity, \
    aggregate_ancestor, normalize, to_binary_path

verbose = False


class Policy(object):
    def __init__(self, args):
        self.wrong_node_reward = args.wrong_reward
        self.state = None
        self.level = 0

        self.inter_mode = args.inter_mode
        self.pred_para = args.pred_para
        if self.inter_mode == 'naive':
            print(f"[Policy] Use naive intermediate statement generation.")
        elif self.inter_mode == 'hybrid':
            print(f"[Policy] Use hybrid intermediate statement generation (long sentence).")
        else:
            raise NotImplementedError

        if args.bleu_reward:
            self.bleu = BLEU(effective_order=True)
            self.useBLEU = True
            print(f"[Policy] Use BLEU reward.")
        else:
            self.useBLEU = False
            print(f"[Policy] Use naive reward.")

        self.sample_gold = 1
        self.epsilon = 1.0  # factor of scheduled sampling

        self.repetition = 0
        self.rep_total = 0.00001

        self.reward_per_node = 0
        self.reward_num = 0

        self.gold_true = 0  # count the number of gold gets higher score, for analysis
        self.only_binary_total = 0

    def init_state(self, candidate_name, evidence):
        # use candidate name to represent state, e.g. [(sent1, sent2), (sent1, sent3), (sent2, sent3)]
        self.state = [{'action': candidate_name.reshape(-1, 2).tolist(), 'all_sents': evidence,
                       'choice': [], 'prob': None, 'used': []}]
        self.level = 0
        self.sample_gold = np.random.choice([0, 1], p=[1 - self.epsilon, self.epsilon])
        if verbose:
            print("========", self.state[0]['action'])

    def to_string(self):
        print("###action0:", self.state[0]['action'])
        print("###all_sents: ", self.state[0]['all_sents'])
        print("###choice0:", self.state[0]['choice'])

    def set_epsilon(self, eps):
        assert 0 < eps <= 1, "epsilon not valid: " + str(eps)
        self.epsilon = eps
        print("[Policy] epsilon set as", self.epsilon)
        print("[Sent Repetition]: %4f, (%d / %d)" % (self.repetition / self.rep_total,
                                                     self.repetition, int(self.rep_total)))

        self.repetition = 0
        self.rep_total = 0.00001
        if self.inter_mode == 'model':
            self.save_inter_cache()

    def update_state_get_new_nodes_offset(self, index, prob, offset, state_index, test=False, K_test=1):
        r"""
        update state and keep track of each decision, using offset of prob_lst
        :param test: test mode, only one beam
        :param index: chosen action index
        :param prob: probability across whole batch
        :return: update policy state and give incremental inputs ids
        """

        new_state = []
        incremental_texts = []
        prev_state_indexes = []
        prev_action_indexes = []
        prev_feat_indexes = []
        longest_index = 0
        NEED_PAD = False

        for i, idx in enumerate(index):
            one_incremental_texts = []
            prev_state_idx = state_index[i]
            prev_state_indexes.append(prev_state_idx)

            prev_cand_name = self.state[prev_state_idx]['action']
            prev_all_sents = copy.deepcopy(self.state[prev_state_idx]['all_sents'])
            prev_used = copy.deepcopy(self.state[prev_state_idx]['used'])

            idx -= sum(offset[:prev_state_idx])
            choice_prob = prob[prev_state_idx][idx]

            if idx >= len(prev_cand_name):
                print("idx larger than prev candidates len", self.state[prev_state_idx])
                print("idx:", idx)

            choice = prev_cand_name[idx]
            for cand in choice:
                if cand not in set(prev_used):
                    prev_used.append(cand)
            prev_action_indexes.append(idx)

            # use specific index to extract related feature
            # 要从上一轮保留哪些feature，删除已经使用过的句子对，需要保留的feature按照new_index list进行存储
            new_index = []
            for j, cand_name in enumerate(prev_cand_name):
                if cand_name[0] in set(choice) or cand_name[1] in set(choice):
                    continue
                else:
                    new_index.append(j)

            prev_feat_indexes.append(new_index)

            if len(new_index) > longest_index:
                longest_index = len(new_index)

            new_cand_name = np.asarray(prev_cand_name)[new_index].tolist()
            new_inter_name = 'int' + str(self.level + 1)

            # generate intermediate conclusion
            sent1 = normalize(prev_all_sents[choice[0]])
            sent2 = normalize(prev_all_sents[choice[1]])

            if self.inter_mode == 'model':
                new_inter_text = self.deduction.get_intermediate_text(sent1, sent2)
            elif self.inter_mode == 'naive':
                new_inter_text = get_intermediate_sentence(sent1, sent2)
            else:
                model_inter_text = self.deduction.get_intermediate_text(sent1, sent2)
                new_inter_text = "According to " + sent1.strip('.') + ", and " \
                                 + sent2.strip('.') + ", we can infer that " \
                                 + model_inter_text.strip(' ')

            if test:
                used_set = set(prev_used)
            else:
                used_set = set(choice)
            for k in prev_all_sents.keys():
                if k not in used_set:
                    new_cand_name.append([k, new_inter_name])
                    one_incremental_texts.append([prev_all_sents[k], new_inter_text])

            incremental_texts.append(one_incremental_texts)
            prev_all_sents[new_inter_name] = new_inter_text

            if self.state[prev_state_idx]['prob'] is not None:
                new_prob = torch.cat([self.state[prev_state_idx]['prob'], choice_prob.unsqueeze(0)], dim=0)
            else:
                new_prob = choice_prob.unsqueeze(0)

            new_state.append({'action': new_cand_name,
                              'all_sents': prev_all_sents,
                              'choice': self.state[prev_state_idx]['choice'] + [choice],
                              'prob': new_prob,
                              'used': prev_used
                              })

        self.state = new_state
        self.level += 1
        if verbose:
            print("@level:", self.level, "samplegold:", self.sample_gold)
            for i in range(len(new_state)):
                print("@State", i, )
                print("@action:", new_state[i]['action'])
                print("@choice:", new_state[i]['choice'])
                print("@used:", new_state[i]['used'])

        return prev_state_indexes, prev_action_indexes, prev_feat_indexes, incremental_texts

    def get_greedy_action(self, prob):
        # greedy action, used in inference
        prob = prob.contiguous().detach().cpu().numpy().reshape(-1)
        actions = [np.argmax(prob)]
        return torch.Tensor(actions).to(torch.int64)

    def get_random_action(self, prob, K):
        # get K (beam size) actions according to prob, monte-carlo sampling
        assert len(prob) > 0, "When getting actions, prob len = 0"
        actions = []
        for i, one_prob in enumerate(prob):
            one_action = np.random.choice([0, 1], p=[1 - one_prob, one_prob])
            if one_action == 1:
                actions.append(i)
        if len(actions) > K:
            actions = random.sample(actions, K)

        while len(actions) < K <= len(prob):
            random_k = random.randint(0, len(prob) - 1)
            if random_k not in actions:
                actions.append(random_k)

        actions = sorted(actions)
        if len(prob) >= K:
            assert len(actions) == K, "num of actions " + str(len(actions)) + " not equal " + str(K)
        else:
            # num of actions is less than K
            actions = K * [0]

        return torch.Tensor(actions).to(torch.int64)

    def get_beam_search_action(self, prob, K):
        """
        beam search with accumulated probability, keep K beams during inference, instead of K branches
        :param prob: probability distribution [#action * K]
        :param K: beam size
        :return: action index
        """

        flat_prob = np.array([[p.detach().cpu() for p in sublist] for sublist in prob])

        assert len(self.state) == len(flat_prob), f"self.state: {len(self.state)}, prob: {len(prob)}"

        all_prob = []
        for i in range(len(self.state)):
            if self.state[i]['prob'] is None:
                prev_prob = np.array([1])
            else:
                prev_prob = self.state[i]['prob'].detach().cpu().numpy()

            one_state_prob = [np.prod(np.exp(prev_prob)) * flat_prob[i]]

            all_prob.extend(one_state_prob)
        all_prob = np.array(all_prob).reshape(-1)

        actions = np.argsort(all_prob)[-K:]
        actions = np.flip(actions).copy().tolist()
        if len(all_prob) < K:
            # actions.extend((K - len(actions)) * [np.argmax(prob)])
            actions = actions + (K - len(actions)) * [np.argmax(all_prob)]
        assert len(actions) == K, f"action len {len(actions)} not equal beam size K."

        return torch.Tensor(actions).to(torch.int64)


    def get_action_scheduled_sampling_hybrid(self, prob, K, gold, source):
        r"""
        sample actions with scheduled sampling
        :param prob: probability distribution
        :param K: beam size
        :param gold: choose gold step
        :return: actions
        """
        if self.sample_gold == 0:
            return self.get_random_action(prob, K)
        else:
            self.rep_total += 1
            actions = []
            one_state = self.state[0]
            for i, step in enumerate(one_state['action']):
                if set(step) == set(gold):
                    actions.append(i)
                    break
            if len(actions) != 1:
                self.repetition += 1  # encounter steps that need used sentences before
            prob = prob.contiguous().detach().cpu().numpy().reshape(-1)
            actions.append(np.argmax(prob))
            if len(actions) < K:
                actions = K * [np.argmax(prob)]

            return torch.Tensor(actions[:K]).to(torch.int64)

    def get_action_scheduled_sampling(self, prob, K, gold):
        r"""
        sample actions with scheduled sampling
        :param prob: probability distribution
        :param K: beam size
        :param gold: choose gold step
        :return: actions
        """
        flat_prob = np.array([p.detach().cpu() for sublist in prob for p in sublist])

        if self.sample_gold == 0:
            return self.get_random_action(flat_prob, K)
        else:
            self.rep_total += 1
            actions = []
            one_state = self.state[0]
            for i, step in enumerate(one_state['action']):
                if set(step) == set(gold):
                    actions.append(i)
                    break
            if len(actions) != 1:
                self.repetition += 1  # encounter steps that need used sentences before
                actions.append(np.argmax(flat_prob))
                if len(actions) < K:
                    actions = K * [np.argmax(flat_prob)]
            else:
                actions = K * actions

            return torch.Tensor(actions[:K]).to(torch.int64)

    def get_stepwise_reward_align_accu_log(self, gold, pred, discount):
        # in accordance with steps-F1 metric, assign positive reward to steps with consistent ancestors
        # accumulate the reward along the trajectory, i.e., each step considers all the reward after it

        aggre_leaves = aggregate_ancestor(gold)
        pred_leaves = {}
        pred_map_gold_inter = {}
        each_step_reward = []
        allcorrect = True

        total = 0
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
            gold_step = gold[map_gold]

            if set(gold_step) == set(aligned_pred_step):
                each_step_reward.append(1)
            else:
                each_step_reward.append(self.wrong_node_reward)
                allcorrect = False

        # accumulate reward for each step, instead of for each path
        acc_stepwise_reward = []
        ac_r = 0
        for i, r in enumerate(each_step_reward):
            coefficient = [discount ** ii for ii in range(i + 1)]
            acc_stepwise_reward.append(sum(coefficient) * r + ac_r)

        return acc_stepwise_reward

    def get_stepwise_reward(self, step_source, intermediate, args):
        """
        Assign cumulative reward for each single step.
        """
        # return reward and the best path, the larger the better
        reward_lst = []
        if len(self.state) == 0:
            print("&&&", step_source)
            print("&ss:", self.sample_gold)
        best_path = self.state[0]['choice']
        best_reward = -100
        for i, st in enumerate(self.state):
            choice = st['choice']
            if verbose:
                print(f"state:, {st['action']};\nchoice:, {st['choice']};\ngold:, {step_source}")
            cur_reward = self.get_stepwise_reward_align_accu_log(step_source, choice, args.discount)
            reward_lst.append(cur_reward)
            if sum(cur_reward) > best_reward:
                best_path = st['choice']
        return reward_lst, best_path

    def inference(self, verbose=False):
        # during inference, find best path according to probability instead of rewards
        best_prob = -1
        best_path = self.state[0]['choice']
        for i, one_state in enumerate(self.state):
            one_prob = torch.prod(one_state['prob'], 0)
            if one_prob > best_prob:
                best_path = one_state['choice']
                best_prob = one_prob

        return best_prob, best_path

    def get_path_reward_align_accu_log(self, gold, pred, state_idx, discount):
        # in accordance with steps-F1 metric, assign positive reward to steps with consistent ancestors
        # accumulate the reward along the trajectory, i.e., each step considers all the reward after it

        aggre_leaves = aggregate_ancestor(gold)
        pred_leaves = {}
        pred_map_gold_inter = {}
        each_step_reward = []

        total = 0
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
            gold_step = gold[map_gold]

            if set(gold_step) == set(aligned_pred_step):
                each_step_reward.append(1)
            else:
                each_step_reward.append(self.wrong_node_reward)
                allcorrect = False
        only_reward = []
        for i, r in enumerate(each_step_reward):
            one_r = r
            for j in range(i + 1, len(each_step_reward)):
                one_r += each_step_reward[j] * (discount ** (j - i))
            total += self.state[state_idx]['prob'][i] * one_r

            self.reward_per_node += one_r
            self.reward_num += 1

            only_reward.append(one_r)

        return total

    def inference_get_reward(self, step_source, args):
        # during inference, find best path according to probability instead of rewards
        best_prob = float('-inf')
        best_path = None
        best_all_sents = None
        best_state_idx = -1
        for i, one_state in enumerate(self.state):
            # sum of log prob
            one_prob = torch.sum(one_state['prob'])

            # one_prob = one_state['prob'][-1]
            if one_prob > best_prob:
                best_path = one_state['choice']
                best_prob = one_prob
                best_all_sents = one_state['all_sents']
                best_state_idx = i

        if self.reward_num > 0:
            self.reward_per_node = 0
            self.reward_num = 0

        # calculate reward for best path
        best_reward = self.get_path_reward_align_accu_log(step_source, best_path, best_state_idx, args.discount)

        return best_prob, best_path, best_all_sents, best_reward.detach().item()


    def save_inter_cache(self):
        save_cache_path = None
        self.deduction.save_inter_cache(save_cache_path)
