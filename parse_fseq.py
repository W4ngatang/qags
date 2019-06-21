""" Extract generations from fairseq outputs """

import re
import json
from collections import defaultdict

def filter_line(line):
    """ Detect if actually a line that we care about """
    return re.match(r'(S|T|H|P)-[0-9]+\t', line) is None

def parse_generation(data_file, out_file):
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
            text = line[1].strip()
            key = 'source' if line_t == 'S' else 'target'
            data[ex_idx][key] = text
        elif line_t == 'H':
            assert len(line) == 3
            score = float(line[1])
            text = line[2].strip()
            data[ex_idx]["hypotheses"].append((text, score))
        else: # probabilities
            continue

    with open(out_file, 'w', encoding='utf-8') as out_fh:
        for datum_idx, datum in data.items():
            out_fh.write(f"{json.dumps({datum_idx: datum})}\n")

data_file = "/checkpoint/wangalexc/fairseq/06-17-2019/summaries.out"
out_file = "/private/home/wangalexc/projects/qags/data/summaries.jsonl"
parse_generation(data_file, out_file)

data_file = "/checkpoint/wangalexc/fairseq/06-18-2019/questions-cnndm.out"
out_file = "/private/home/wangalexc/projects/qags/data/questions.jsonl"
parse_generation(data_file, out_file)
