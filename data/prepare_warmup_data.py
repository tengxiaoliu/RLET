import jsonlines
import json
import copy
import os

# process task 1 data
data_dir = 'data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
generated_dir = 'data/warmup/task1'
os.makedirs(generated_dir, exist_ok=True)
data_type = ['train', 'dev', 'test']

def clean_sample(sample):
    exceptions = 0
    generated = []
    case_id = sample['id']
    facts = sample['meta']['triples']
    inters = sample['meta']['intermediate_conclusions']
    all_sents = copy.deepcopy(facts)
    all_sents.update(inters)

    proof = sample['proof'].split("; ")[:-1]

    # iterate each step, construct a sentence pair selection data for each step
    for step_idx, step in enumerate(proof):
        entry = {'id': case_id, 'step_id': step_idx}
        src, tgt = step.split(':')[0].split(' -> ')
        src_lst = list(set(src.split(' & ')))  # remove duplicated sents
        if tgt == 'hypothesis':
            tgt = sample['meta']['hypothesis_id']
        for s in src_lst:
            if s not in facts:
                facts[s] = all_sents[s]
                exceptions += 1
        entry['sentences'] = copy.deepcopy(facts)
        entry['chosen'] = src_lst
        entry['conclusion'] = {tgt: inters[tgt]}
        entry['step'] = step.replace('hypothesis', sample['meta']['hypothesis_id'])
        entry['hypothesis'] = sample['hypothesis']
        entry['question'] = sample['question']
        entry['answer'] = sample['answer']

        # update current facts set state
        for s in src_lst:
            facts.pop(s)
        facts[tgt] = inters[tgt]
        generated.append(entry)
    return generated, exceptions


if __name__ == "__main__":

    for key in data_type:
        generated_data = []
        with open(os.path.join(data_dir, f"{key}.jsonl"), "r+", encoding="utf8") as f:
            total_except = 0
            for item in jsonlines.Reader(f):
                samples, num_except = clean_sample(copy.deepcopy(item))
                generated_data.extend(samples)
                total_except += num_except
            print("[Done] Generated %d data items on %s. \nEncountered %d exceptions."
                  % (len(generated_data), key, total_except))

        with open(os.path.join(generated_dir, f"{key}.jsonl"), mode='w', encoding='utf8') as outfile:
            json.dump(generated_data, outfile)

    print("Generated data in", generated_dir)
