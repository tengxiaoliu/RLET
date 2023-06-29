import argparse
import jsonlines
import json
import copy
import os

# process filtered data
ori_data_dir = 'data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
generated_dir = os.path.join(ori_data_dir, "filter")

task2_dirs = {
    'dev': 'outputs/examples/task2_filter/dev.jsonl',
    'test': 'outputs/examples/task2_filter/test.jsonl'
}

task3_dirs = {
    'dev': 'outputs/examples/task3_filter/dev.jsonl',
    'test': 'outputs/examples/task3_filter/test.jsonl'
}

os.makedirs(generated_dir, exist_ok=True)


def add_filter(ori_in_file, fil):
    exception = 0
    ori = copy.deepcopy(ori_in_file)
    assert ori['id'] == fil['id'], "id not consistent! id: " + ori['id'] + ", filter: " + fil['id']

    facts = ori['meta']['triples']
    filtered_keys = fil['pred']

    all_keys = copy.deepcopy(set(facts.keys()))
    for k in all_keys:
        if k not in set(filtered_keys):
            facts.pop(k)
    if len(facts) <= 1:
        exception += 1
        ori['meta']['triples'] = copy.deepcopy(ori_in_file['meta']['triples'])

    return ori, exception


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default='top5', type=str)
    args = parser.parse_args()

    if args.filter == 'task2':
        filter_dirs = task2_dirs
    elif args.filter == 'task3':
        filter_dirs = task3_dirs
        ori_data_dir = 'data/entailment_trees_emnlp2021_data_v3/dataset/task_3'
        generated_dir = os.path.join(ori_data_dir, "filter")
    else:
        raise NotImplementedError

    for key in ['dev', 'test']:
        generated_data = []
        ori_items = []
        filter_items = []
        filter_dir = filter_dirs[key]
        exception = 0

        with open(os.path.join(ori_data_dir, f"{key}.jsonl"), "r+", encoding="utf8") as ori_file:
            for item in jsonlines.Reader(ori_file):
                ori_items.append(item)

        print("[Loading] filter file from ", os.path.split(filter_dir)[-1])

        with open(filter_dir, "r+", encoding="utf8") as filter_file:
            filter_items = json.load(filter_file)

        assert len(ori_items) == len(filter_items), "length not equal" + str(len(filter_items))
        for ori_in_file, fil in zip(ori_items, filter_items):
            filtered_data, exc = add_filter(ori_in_file, fil)
            generated_data.append(filtered_data)
            exception += exc

        with open(os.path.join(generated_dir, f"{key}.jsonl"), mode='w', encoding='utf8') as outfile:
            json.dump(generated_data, outfile)

        print("[Filter] Encountered %d exceptions." % (exception))
        print("[Done] Generated %d data items at %s." % (len(generated_data),
                                                         os.path.join(generated_dir, f"{key}.jsonl")))
