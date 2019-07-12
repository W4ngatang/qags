""" Various data processing utility functions """

import re
import json
import copy
import random
import itertools
from collections import defaultdict

def filter_line_fseq(line):
    """ Detect if actually a line that we care about
    from a fairseq log
    """
    return re.match(r'(S|T|H|P)-[0-9]+\t', line) is None

def write_jsonl(data, out_file):
    """ Write a dictionary to out_file as a jsonl """
    with open(out_file, 'w', encoding='utf-8') as out_fh:
        for datum_idx, datum in data.items():
            out_fh.write(f"{json.dumps({datum_idx: datum})}\n")

def write_text(data, out_file):
    """ Write out an iterable of texts, one text per line """
    with open(out_file, 'w', encoding='utf-8') as out_fh:
        for datum in data:
            out_fh.write(f"{datum}\n")

def process(text):
    """ Strip [CLS] token from text """
    return text.replace("[CLS]", "").strip()

def print_samples(data, n_samples=5):
    """ Print samples from dictionary """
    samples = random.sample(list(data.values()), n_samples)
    for sample in samples:
        print(f"Source: {sample['src']}")
        sample["gen"].sort(key=lambda x: x[1], reverse=True)
        for hyp in sample["gen"]:
            print(f"\tHyp: {hyp[0]}")

def parse_generation(data_file):
    """ Parse data_file (fairseq log) for the actual generations.

    returns:
        - data: dict mapping example indices to dictionary with keys
            'src', 'trg', and 'gen', where the latter is a list
    """
    with open(data_file, encoding='utf-8') as data_fh:
        all_lines = data_fh.readlines()

    data = defaultdict(lambda: defaultdict(dict))
    for line in all_lines:
        if filter_line_fseq(line):
            continue

        line = line.split('\t')
        line_t, ex_idx = line[0].split('-')
        ex_idx = int(ex_idx)
        if "gen" not in data[ex_idx]:
            data[ex_idx]["gen"] = []
        data[ex_idx]["id"] = ex_idx

        assert line_t in ['S', 'T', 'H', 'P']
        if line_t in ['S', 'T']:
            assert len(line) == 2
            text = process(line[1])# .strip()
            key = 'src' if line_t == 'S' else 'trg'
            data[ex_idx][key] = text
        elif line_t == 'H':
            assert len(line) == 3
            score = float(line[1])
            text = process(line[2]) #.strip()
            data[ex_idx]["gen"].append((text, score))
        else: # probabilities
            continue

    return data

def swap_fields(data, swap_d):
    """ Swap names of keys in dictionary """
    new_data = defaultdict(lambda: defaultdict(dict))
    for datum_idx, datum in data.items():
        new_datum = copy.deepcopy(datum)
        for k, v in swap_d.items():
            if k in new_datum:
                del new_datum[k]
            if v in new_datum:
                del new_datum[v]
        for k, v in swap_d.items():
            new_datum[v] = datum[k]
        new_data[datum_idx] = new_datum
    return new_data

def format_squad(raw_data, context="src"):
    """ Format data into SQuAD format.

    args:
        - raw_data: dictionary mapping example indices to
            dicts of 'src', 'trg', 'gen' (list)
        - context: name of the text field in raw_data to use as the 'source document'

    returns:
        - data: SQuAD formatted data
    """
    assert context in ["src", "trg", "gen"]
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
        data.append({"paragraphs": [datum],
                     "title": dummy_title,
                    })

    return {"data": data}


