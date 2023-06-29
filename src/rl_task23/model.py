import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from fastNLP import print

from policy import Policy
from utils import get_tokenize_inputs

LOG = False

class EBRLModel(nn.Module):
    def __init__(self, args, ptm=None):
        super().__init__()
        # init with trained parameters
        config = AutoConfig.from_pretrained(args.policy_model_config)
        if ptm is None:
            self.ptm = AutoModel.from_config(config)
        else:
            self.ptm = ptm

        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.ptm.config.hidden_size, 1)
        self.policy = Policy(args)
        self.args = args

    def train_step(self, input_ids=None, attention_mask=None, token_type_ids=None, evidence=None,
                   intermediate=None, candidate_name=None, step_source=None, hypothesis=None):

        # ========== 1. Forward: save input_ids and stepwise reward ==========
        with torch.no_grad():
            all_step_inputs = []
            all_step_offsets = []
            all_step_actions = []
            all_step_states = []
            all_step_embeddings = []
            all_step_chosen_embeddings = []

            level = 0
            evidence = evidence[0]
            step_source = np.array(step_source)
            candidate_name = np.array(candidate_name)
            step_source = np.squeeze(step_source)

            if len(step_source.shape) == 1 and isinstance(step_source[0], str):
                step_source = np.expand_dims(step_source, axis=0)

            # ===== Task 2 =====
            level_limit = len(step_source)
            step_source = np.append(step_source, [['int' + str(level_limit), 'end']], axis=0)
            level_limit += 1

            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])

            ori_inputs = {"input_ids": input_ids,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids}

            cur_embedding = self.ptm(**ori_inputs)[0][:, 0, :]
            inc_inputs, inc_offsets = None, None

            self.policy.init_state(candidate_name, evidence)
            prev_state_index, prev_action_index, prev_feat_index = None, None, None
            prev_embedding, prev_inputs, prev_offset = None, ori_inputs, None

            while level < level_limit:
                # encode incremental sentence pairs
                if inc_inputs is not None:
                    # get CLS embedding
                    inc_embedding = self.ptm(**inc_inputs)[0][:, 0, :].contiguous().squeeze()  # inc_num * emb_size
                    inc_embedding = [inc_embedding[sum(inc_offsets[:i]): sum(inc_offsets[:i + 1])]
                                     for i in range(len(inc_offsets))]

                    # prev_state_index与prev_action_index应该是一一对应，代表action来自哪个state
                    assert prev_state_index is not None and prev_action_index is not None, "while loop not consistent."
                    assert len(prev_state_index) == len(prev_action_index), "action and state not equal."

                    cur_embedding = []
                    one_all_step_inputs = []
                    # 遍历state index
                    for i, state_idx in enumerate(prev_state_index):
                        # 如果当前state index在上一步没有剩余的embedding，则直接用新添加的embedding

                        emb_l = sum(inc_offsets[:i])
                        emb_r = sum(inc_offsets[:i + 1])
                        one_inc_inputs = get_inputs_subset(inc_inputs, emb_l, emb_r)

                        if len(prev_feat_index[i]) == 0:
                            cur_embedding.append(inc_embedding[i])
                            # save inputs
                            one_all_step_inputs.append(one_inc_inputs)

                        # 否则，进行向量拼接，用来自上一步的剩余prev_embedding拼上新添加的embedding
                        else:
                            cur_embedding.append(torch.cat([prev_embedding[state_idx][prev_feat_index[i]],
                                                            inc_embedding[i]], dim=0))
                            # save inputs

                            if type(prev_inputs) == dict:
                                prev_inputs = [prev_inputs]
                            one_prev_inputs = get_inputs_subset_index(prev_inputs[state_idx], prev_feat_index[i])
                            one_all_step_inputs.append(cat_inputs(one_prev_inputs, one_inc_inputs))
                        assert one_all_step_inputs[-1]['input_ids'].shape[0] == cur_embedding[-1].shape[0], \
                            f"inputs shape {one_all_step_inputs[-1]['input_ids'].shape} != emb{cur_embedding[-1].shape}"

                    cur_emb_offset = [len(embs) for embs in cur_embedding]

                if level == 0:
                    prev_inputs = ori_inputs
                    prev_embedding = cur_embedding.unsqueeze(0)
                    cur_emb_offset = [prev_embedding.shape[1]]
                    one_all_step_inputs = ori_inputs
                    all_step_cur_embs = [cur_embedding]

                else:
                    # emb.squeeze() to a flat tensor
                    prev_inputs = one_all_step_inputs
                    prev_embedding = cur_embedding
                    all_step_cur_embs = [torch.stack([emb for emb in sublist]) for sublist in cur_embedding]
                    cur_embedding = torch.stack([emb for sublist in cur_embedding for emb in sublist])

                # save offsets
                all_step_offsets.append(cur_emb_offset)
                all_step_inputs.append(one_all_step_inputs)
                all_step_embeddings.extend(all_step_cur_embs)
                cur_no_drop_emb = cur_embedding
                cur_embedding = self.dropout(cur_embedding)
                logits = self.classifier(cur_embedding).view(-1)
                # 对每一beam根据cur_emb_offset进行softmax
                log_prob_lst = []
                prob_lst = []
                for i in range(len(cur_emb_offset)):
                    one_log_prob = F.log_softmax(logits[sum(cur_emb_offset[:i]): sum(cur_emb_offset[:i + 1])], dim=-1)
                    log_prob_lst.append(one_log_prob)
                    one_prob = torch.exp(one_log_prob)
                    prob_lst.append(one_prob)
                # sample actions according to the prob distribution
                action_index = self.policy.get_action_scheduled_sampling(prob_lst, self.args.K, step_source[level])
                state_index = []

                for action_idx in action_index:
                    state_ptr = 0
                    while action_idx > sum(cur_emb_offset[:state_ptr]) \
                            and action_idx >= sum(cur_emb_offset[:state_ptr + 1]):
                        state_ptr += 1
                    assert sum(cur_emb_offset[:state_ptr]) <= action_idx < sum(cur_emb_offset[:state_ptr + 1]), \
                        f"action_idx: {action_idx}, state_ptr: {state_ptr}"
                    state_index.append(state_ptr)

                assert len(action_index) == self.args.K, f"Number of selections is not consistent with beam size {self.args.K}"
                # update the state by the sampled action
                # incremental_texts长度不一定一样
                prev_state_index, prev_action_index, prev_feat_index, incremental_texts, _ = \
                    self.policy.update_state_get_new_nodes_offset(action_index, prob_lst, cur_emb_offset,
                                                                  state_index, test=False)

                level += 1
                all_step_actions.append(prev_action_index)
                all_step_states.append(prev_state_index)
                for i, action_idx in enumerate(prev_action_index):
                    all_step_chosen_embeddings.append(all_step_cur_embs[all_step_states[-1][i]][action_idx])
                # 对新生成的句子pair进行encode
                if level < len(step_source):
                    inc_offsets = [len(inc) for inc in incremental_texts]
                    flat_incremental_texts = [text for sublist in incremental_texts for text in sublist]
                    inc_inputs = get_tokenize_inputs(flat_incremental_texts, self.args, hypothesis)

            # get stepwise reward_lst: beam_size * stepwise reward list
            reward_lst, best_path = self.policy.get_stepwise_reward(step_source, intermediate, self.args)

        # ========== 2. Stepwise backward: loss.backward() ==========

        total_loss = []

        stepwise_log_prob = [[], []]
        stepwise_embeddings = []
        stepwise_chosen_embeddings = []

        loss_num = len(all_step_actions) * self.args.K
        for step_idx in range(len(all_step_actions)):
            if step_idx == 0:
                one_inputs = all_step_inputs[0]
                one_embedding = self.ptm(**one_inputs)[0][:, 0, :].contiguous()
                stepwise_embeddings.append(one_embedding)
                one_embedding = self.dropout(one_embedding)
                logits = self.classifier(one_embedding).view(-1)
                log_prob = F.log_softmax(logits, dim=-1)
                prob = torch.exp(log_prob)
                loss = []

                for beam_i, action_idx in enumerate(all_step_actions[0]):
                    one_reward = reward_lst[beam_i][0]
                    if LOG:
                        one_loss = - log_prob[action_idx] * one_reward / loss_num
                    else:
                        one_loss = - prob[action_idx] * one_reward / loss_num

                    loss.append(one_loss)
                    total_loss.append(one_loss.detach())

                    stepwise_log_prob[beam_i].append(log_prob[action_idx].detach())
                    stepwise_chosen_embeddings.append(stepwise_embeddings[-1][action_idx])

                # 如果是最后一个loss，则不在此处backward，而是交给trainer进行backward
                if step_idx == len(all_step_actions) - 1:
                    loss = torch.stack(loss)
                    loss = loss.sum()
                    continue
                loss = torch.stack(loss).sum()
                loss.backward()
            else:
                for beam_i, action_idx in enumerate(all_step_actions[step_idx]):

                    one_inputs = all_step_inputs[step_idx][all_step_states[step_idx][beam_i]]
                    one_embedding = self.ptm(**one_inputs)[0][:, 0, :].contiguous()
                    one_embedding = one_embedding.view(-1, one_embedding.shape[-1])
                    stepwise_embeddings.append(one_embedding)
                    one_embedding = self.dropout(one_embedding)
                    logits = self.classifier(one_embedding).view(-1)
                    log_prob = F.log_softmax(logits, dim=-1)
                    prob = torch.exp(log_prob)

                    one_reward = reward_lst[beam_i][step_idx]
                    if LOG:
                        loss = - log_prob[action_idx] * one_reward / loss_num
                    else:
                        loss = - prob[action_idx] * one_reward / loss_num

                    total_loss.append(loss.detach())

                    stepwise_log_prob[beam_i].append(log_prob[action_idx].detach())
                    stepwise_chosen_embeddings.append(stepwise_embeddings[-1][action_idx])

                    if step_idx == len(all_step_actions) - 1 and beam_i == len(all_step_actions[step_idx]) - 1:
                        continue
                    loss.backward()

        return {'loss': loss, "pred": best_path, 'target': step_source}

    def evaluate_step(self, input_ids=None, attention_mask=None, token_type_ids=None, evidence=None,
                      intermediate=None, candidate_name=None, step_source=None, hypothesis=None):

        level = 0
        evidence = evidence[0]
        step_source = np.array(step_source)
        candidate_name = np.array(candidate_name)
        if not self.args.task3:
            step_source = np.squeeze(step_source)
            if len(step_source.shape) == 1 and isinstance(step_source[0], str):
                step_source = np.expand_dims(step_source, axis=0)

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])

        ori_inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids}
        cur_embedding = self.ptm(**ori_inputs)[0][:, 0, :]
        inc_inputs, inc_offsets = None, None

        self.policy.init_state(candidate_name, evidence)
        prev_state_index, prev_action_index, prev_feat_index = None, None, None
        prev_embedding, prev_inputs, prev_offset = None, ori_inputs, None

        level_limit = max(1, len(evidence.keys()) + 2)

        while level < level_limit:
            # encode incremental sentence pairs
            if inc_inputs is not None:
                # get CLS embedding
                inc_embedding = self.ptm(**inc_inputs)[0][:, 0, :].contiguous().squeeze()  # inc_num * emb_size
                inc_embedding = inc_embedding.view(-1, inc_embedding.shape[-1])
                inc_embedding = [inc_embedding[sum(inc_offsets[:i]): sum(inc_offsets[:i + 1])]
                                 for i in range(len(inc_offsets))]

                # prev_state_index与prev_action_index应该是一一对应，代表action来自哪个state
                assert prev_state_index is not None and prev_action_index is not None, "while loop not consistent."
                assert len(prev_state_index) == len(prev_action_index), "action and state not equal."

                cur_embedding = []
                # 遍历state index
                for i, state_idx in enumerate(prev_state_index):

                    if len(prev_feat_index[i]) == 0:
                        cur_embedding.append(inc_embedding[i])
                    # 否则，进行向量拼接，用来自上一步的剩余prev_embedding拼上新添加的embedding
                    else:
                        cur_embedding.append(torch.cat([prev_embedding[state_idx][prev_feat_index[i]],
                                                        inc_embedding[i]], dim=0))
                cur_emb_offset = [len(embs) for embs in cur_embedding]

            if level == 0:
                prev_embedding = cur_embedding.unsqueeze(0)
                cur_emb_offset = [prev_embedding.shape[1]]

            else:
                prev_embedding = cur_embedding
                cur_embedding = torch.stack([emb for sublist in cur_embedding for emb in sublist])

            cur_embedding = self.dropout(cur_embedding)
            logits = self.classifier(cur_embedding).view(-1)
            # 对每一beam根据cur_emb_offset进行softmax
            log_prob_lst = []
            prob_lst = []
            for i in range(len(cur_emb_offset)):
                one_log_prob = F.log_softmax(logits[sum(cur_emb_offset[:i]): sum(cur_emb_offset[:i + 1])], dim=-1)
                log_prob_lst.append(one_log_prob)
                one_prob = torch.exp(one_log_prob)
                prob_lst.append(one_prob)

            # sample actions according to the prob distribution with greedy beam
            action_index = self.policy.get_beam_search_action(prob_lst, self.args.K_test)
            state_index = []
            for action_idx in action_index:
                state_ptr = 0
                while action_idx > sum(cur_emb_offset[:state_ptr]) \
                        and action_idx >= sum(cur_emb_offset[:state_ptr + 1]):
                    state_ptr += 1
                assert sum(cur_emb_offset[:state_ptr]) <= action_idx < sum(cur_emb_offset[:state_ptr + 1]), \
                    f"action_idx: {action_idx}, state_ptr: {state_ptr}, sum: {sum(cur_emb_offset[:state_ptr])}"
                state_index.append(state_ptr)

            assert len(action_index) == self.args.K_test, f"Number of selections is not consistent with beam size " \
                                                          f"{self.args.K_test}"
            # update the state by the sampled action
            # incremental_texts长度不一定一样
            prev_state_index, prev_action_index, prev_feat_index, incremental_texts, is_end = \
                self.policy.update_state_get_new_nodes_offset(action_index, prob_lst, cur_emb_offset,
                                                              state_index, test=True)

            level += 1
            if is_end:
                break

            # 对新生成的句子pair进行encode
            if level < level_limit:
                inc_offsets = [len(inc) for inc in incremental_texts]
                flat_incremental_texts = [text for sublist in incremental_texts for text in sublist]
                if len(flat_incremental_texts) == 0:
                    break
                inc_inputs = get_tokenize_inputs(flat_incremental_texts, self.args, hypothesis)

        best_prob, best_path, best_all_sents, best_reward = self.policy.inference_get_reward(step_source, self.args)
        best_all_sents['hypothesis'] = hypothesis[0]

        return {'loss': best_prob, "pred": best_path, 'target': step_source,
                'all_sents': best_all_sents, 'reward': best_reward}


class SentencePairModel(nn.Module):
    def __init__(self, pretrained_model, args):
        super().__init__()
        if pretrained_model is None:
            config = AutoConfig.from_pretrained(args.model_config)
            self.ptm = AutoModel.from_config(config)
        else:
            self.ptm = pretrained_model
        self.dropout = nn.Dropout(args.dropout)

        self.classifier = nn.Linear(self.ptm.config.hidden_size, 1)

    def forward(self):
        pass


def get_inputs_subset(inputs, left, right):
    r"""
    get [left:right] of inputs dict
    """

    return {
        key: inputs[key][left: right] for key in inputs.keys()
    }


def get_inputs_subset_index(inputs, indexes):
    r"""
    get [indexes] of inputs dict
    """
    return {
        key: inputs[key][indexes] for key in inputs.keys()
    }

def cat_inputs(dict1, dict2):
    r"""
    concatenate the value of each keys, for inputs
    padding as well
    """

    longest = dict1['input_ids'].shape[-1] if dict1['input_ids'].shape[-1] > dict2['input_ids'].shape[-1] \
        else dict2['input_ids'].shape[-1]
    ret = {}
    for key in dict1.keys():
        value1 = F.pad(dict1[key], pad=(0, longest-dict1[key].shape[-1], 0, 0), mode='constant', value=0)
        value2 = F.pad(dict2[key], pad=(0, longest-dict2[key].shape[-1], 0, 0), mode='constant', value=0)
        ret[key] = torch.cat([value1, value2], dim=0)

    return ret
