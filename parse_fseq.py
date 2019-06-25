""" Extract generations from fairseq outputs """

import re
import json
import random
from collections import defaultdict

import ipdb

N_SAMPLES = 5

def filter_line(line):
    """ Detect if actually a line that we care about """
    return re.match(r'(S|T|H|P)-[0-9]+\t', line) is None

def write_jsonl(data, out_file):
    with open(out_file, 'w', encoding='utf-8') as out_fh:
        for datum_idx, datum in data.items():
            out_fh.write(f"{json.dumps({datum_idx: datum})}\n")

def process(text):
    return text.replace("[CLS]", "").strip()

def print_samples(data, n_samples=5):
    samples = random.sample(list(data.values()), n_samples)
    for sample in samples:
        print(f"Source: {sample['source']}")
        sample["hypotheses"].sort(key=lambda x: x[1], reverse=True)
        for hyp in sample["hypotheses"]:
            print(f"\tHyp: {hyp[0]}")

def parse_generation(data_file):
    with open(data_file, encoding='utf-8') as data_fh:
        all_lines = data_fh.readlines()

    data = defaultdict(lambda: defaultdict(dict))
    for line in all_lines:
        if filter_line(line):
            continue

        line = line.split('\t')
        line_t, ex_idx = line[0].split('-')
        ex_idx = int(ex_idx)
        if "hypotheses" not in data[ex_idx]:
            data[ex_idx]["hypotheses"] = []
        data[ex_idx]["id"] = ex_idx

        assert line_t in ['S', 'T', 'H', 'P']
        if line_t in ['S', 'T']:
            assert len(line) == 2
            text = process(line[1])# .strip()
            key = 'source' if line_t == 'S' else 'target'
            data[ex_idx][key] = text
        elif line_t == 'H':
            assert len(line) == 3
            score = float(line[1])
            text = process(line[2]) #.strip()
            data[ex_idx]["hypotheses"].append((text, score))
        else: # probabilities
            continue

    return data

def format_squad(raw_data, context="source"):
    """ """
    assert context in ["source", "target"]
    qa_idx = 0
    data = []
    for datum_idx, raw in raw_data.items():
        datum = {}
        datum["context"] = raw[context]

        #dummy_title = random.choice(raw[context].split())
        dummy_title = " ".join(raw[context].split()[:5])

        qas = []
        for raw_qa in raw["hypotheses"]:
            qa = {"question": raw_qa[0],
                  "answers": [],
                  "id": qa_idx
                 }
            qa_idx += 1
            qas.append(qa)
        datum["qas"] = qas
        data.append({"paragraph": [datum],
                     "title": dummy_title,
                    })

    return {"data": data}

#data_file = "/checkpoint/wangalexc/fairseq/06-17-2019/summaries.out"
#out_file = "/private/home/wangalexc/projects/qags/data/summaries.jsonl"
#data = parse_generation(data_file, out_file)
#write_jsonl(data, out_file)

#data_file = "/checkpoint/wangalexc/fairseq/06-18-2019/questions-cnndm.out"
#out_file = "/private/home/wangalexc/projects/qags/data/questions.jsonl"
#data = parse_generation(data_file)
#write_jsonl(data, out_file)
#data = format_squad(data, "source")

data_file = "/checkpoint/wangalexc/fairseq/06-18-2019/questions-cnndm.sampling.out"
out_file = "/private/home/wangalexc/projects/qags/data/questions.sampling.jsonl"
data = parse_generation(data_file)
print_samples(data, n_samples=N_SAMPLES)
write_jsonl(data, out_file)

