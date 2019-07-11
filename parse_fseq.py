""" Extract generations from fairseq outputs """

import re
import json
import copy
import random
from collections import defaultdict

import ipdb

N_SAMPLES = 5

def filter_line(line):
    """ Detect if actually a line that we care about """
    return re.match(r'(S|T|H|P)-[0-9]+\t', line) is None

def write_jsonl(data, out_file, swap_fields):
    with open(out_file, 'w', encoding='utf-8') as out_fh:
        for datum_idx, datum in data.items():
            out_fh.write(f"{json.dumps({datum_idx: datum})}\n")

def write_text(data, out_file):
    """ Write out an iterable of sentences """
    with open(out_file, 'w', encoding='utf-8') as out_fh:
        for datum in data:
            out_fh.write(f"{datum}\n")

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
    """ Parse data_file (fairseq log) for the actual generations.

    returns:
        - data: dict mapping example indices to dictionary with keys
            'src', 'trg', and 'gen', where the latter is a list
    """
    with open(data_file, encoding='utf-8') as data_fh:
        all_lines = data_fh.readlines()

    data = defaultdict(lambda: defaultdict(dict))
    for line in all_lines:
        if filter_line(line):
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
    """ """
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

def format_squad(raw_data, context="source", reverse=False):
    """ """
    assert context in ["source", "target", "summaries"]
    if reverse:
        context = "source" if context == "target" else "target"
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

# Extract summaries
def extract_src_trg_gen_from_fseq_log():
    """ Extract """
    data_file = "/checkpoint/wangalexc/fairseq/07-01-2019/cnndm-summaries.out"
    data = parse_generation(data_file)
    import ipdb; ipdb.set_trace()

    for txt_type in ["src", "gen", "trg"]:
        txts = [d[txt_type] for d in data.values() if len(d['gen']) > 0]
        out_file = f"/private/home/wangalexc/projects/qags/data/{txt_type}.txt"
        write_text(txts, out_file)
        print(f"Wrote {len(txts)} texts to {out_file}")


# Extract questions
def extract_questions_and_write_jsonl():
    #data_file = "/checkpoint/wangalexc/fairseq/06-18-2019/questions-cnndm.sampling.out"
    #out_file = "/private/home/wangalexc/projects/qags/data/questions.sampling.jsonl"

    #data_file = "/checkpoint/wangalexc/fairseq/06-27-2019/questions.cnndm-target.out"
    #out_file = "/private/home/wangalexc/projects/qags/data/questions.cnndm-target.jsonl"
    #aux_file = "/private/home/wangalexc/projects/qags/data/questions.cnndm-summaries.jsonl"
    #swap_d = {"source": "target", "target": "source" } # will error as is

    data_file = "/checkpoint/wangalexc/fairseq/07-09-2019/questions-cnndm-summaries-full.out"
    out_file = "/private/home/wangalexc/projects/qags/data/questions.cnndm-summaries.jsonl"
    #aux_file = "/private/home/wangalexc/projects/qags/data/questions.cnndm-targets.jsonl"
    swap_d = {"source": "generation", "target": "source" }

    data = parse_generation(data_file)
    data = swap_fields(data, swap_d)
    write_jsonl(data, out_file, swap_fields)
    print_samples(data, n_samples=N_SAMPLES)
    print(f"Extracted questions from {data_file} and wrote to {out_file}")

# Extract questions and format as SQuAD
def extract_questions_and_write_squad_json():
    question_source = "summaries"
    context = "source"
    #data_file = f"/checkpoint/wangalexc/fairseq/06-27-2019/questions-cnndm-{question_source}.out"
    data_file = f"/checkpoint/wangalexc/fairseq/07-09-2019/questions-cnndm-{question_source}-full.out"
    out_file = f"/private/home/wangalexc/projects/qags/data/questions.cnndm-{question_source}.{context}.json"

    data = parse_generation(data_file)

    swap_d = {"source": "summaries", "target": "source" }
    data = swap_fields(data, swap_d)

    data = format_squad(data, context)
    json.dump(data, open(out_file, "w", encoding="utf-8"))
    print(f"Extracted questions from {data_file} and wrote to {out_file}")

extract_src_trg_gen_from_fseq_log()
#extract_questions_and_write_jsonl()
#extract_questions_and_write_squad_json()
