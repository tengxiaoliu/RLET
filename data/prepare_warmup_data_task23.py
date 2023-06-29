import jsonlines
import json
import re
import os
import copy
import random

# process task 2 data
data_dir = 'data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
generated_dir = 'data/warmup/task23'
os.makedirs(generated_dir, exist_ok=True)
data_type = ['train', 'dev', 'test']


def have_overlap(sent1, sent2):
    stopwords = {'a', 'the', 'is', 'are', 'of', '.', 'kind', 'to', '/', 'on'}
    s1 = set(sent1.strip('.').split()) - stopwords
    s1 = {re.sub(r"^(\w{3,}?)(?:es|s|ing|e|ed)$", r'\1', word) for word in s1}
    s2 = set(sent2.strip('.').split()) - stopwords
    s2 = {re.sub(r"^(\w{3,}?)(?:es|s|ing|e|ed)$", r'\1', word) for word in s2}
    return not s1.isdisjoint(s2)


def clean_sample(sample):
    exceptions = 0
    sum_sents = 0
    generated = []
    case_id = sample['id']
    ori_facts = sample['meta']['triples']
    inters = sample['meta']['intermediate_conclusions']
    facts = copy.deepcopy(ori_facts)
    facts['end'] = '[END]'  # '[END]'
    all_sents = copy.deepcopy(facts)
    all_sents.update(inters)

    proof = sample['proof'].split("; ")[:-1]

    end_step = 'int' + str(len(inters)) + ' & end -> [END]'
    proof.append(end_step)

    # iterate each step, construct a sentence pair selection data for each step
    src_lst = []
    tgt = None
    for step_idx, step in enumerate(proof):
        one_facts = copy.deepcopy(facts)
        # remove last step src
        for s in src_lst:
            if s in one_facts.keys():
                one_facts.pop(s)
        if tgt is not None:
            one_facts[tgt] = inters[tgt]

        entry = {'id': case_id, 'step_id': step_idx}
        src, tgt = step.split(':')[0].split(' -> ')
        src_lst = list(set(src.split(' & ')))  # remove duplicated sentences
        if tgt == 'hypothesis':
            tgt = sample['meta']['hypothesis_id']
        for s in src_lst:
            if s not in one_facts:
                one_facts[s] = all_sents[s]

        entry['chosen'] = src_lst
        entry['step'] = step.replace('hypothesis', sample['meta']['hypothesis_id'])
        entry['hypothesis'] = sample['hypothesis']
        entry['question'] = sample['question']

        # heuristic filter the negative sentences
        keep_sents_keys = copy.deepcopy(src_lst)
        keep_sents_keys.append('end')
        # if one sent has overlap with hypo or one of the chosen sentences, we'll keep it as a negative sample.
        compare_sent = sample['hypothesis'] + ' ' + ' '.join([one_facts[k] for k in src_lst])
        compare_sent = ' '.join([one_facts[k] for k in src_lst])
        fact_keys_lst = list(one_facts.keys())
        random.shuffle(fact_keys_lst)
        for k in fact_keys_lst:
            if have_overlap(one_facts[k], compare_sent):
                keep_sents_keys.append(k)
            keep_sents_keys = list(set(keep_sents_keys))
            if len(keep_sents_keys) >= 10:
                break
        if len(keep_sents_keys) < 3:
            keep_sents_keys += fact_keys_lst[:3]
            # print(keep_sents_keys)

        entry['sentences'] = {k: all_sents[k] for k in keep_sents_keys}
        assert len(entry['sentences']) >= 3, f"len: {len(entry['sentences'])}"
        sum_sents += len(keep_sents_keys)

        generated.append(entry)
    return generated, exceptions, sum_sents


if __name__ == "__main__":

    for key in data_type:
        generated_data = []
        sent_num = 0
        with open(os.path.join(data_dir, f"{key}.jsonl"), "r+", encoding="utf8") as f:
            total_except = 0
            for item in jsonlines.Reader(f):
                samples, num_except, sum_sents = clean_sample(copy.deepcopy(item))
                generated_data.extend(samples)
                total_except += num_except
                sent_num += sum_sents
            print(f"[Done] Generated {len(generated_data)} data items on {key}. \n"
                  f"Avg sent num: {round(sent_num / len(generated_data), 4)}.")

        with open(os.path.join(generated_dir, f"{key}.jsonl"), mode='w', encoding='utf8') as outfile:
            json.dump(generated_data, outfile, indent=4)

    print("Generated data in", generated_dir)


