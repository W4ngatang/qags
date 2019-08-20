""" Extract generations from fairseq outputs """

import os
import re
import json
import copy
import random
import itertools
from collections import defaultdict
from utils import write_data, write_jsonl, write_txt, \
                  process, print_samples, format_squad, \
                  filter_line_fseq, parse_generation, \
                  load_txt, load_json

N_SAMPLES = 5


def extract_src_trg_gen_from_fseq_log():
    """ Extract source ('S'), target ('T'), and hypothesis generations ('H')
    from fseq logs and write each as a text file, one text per line. """

    append_tags = False
    data_file = "/checkpoint/wangalexc/fairseq/08-11-2019/qst.src-subset.cnndm.test.txt"
    data = parse_generation(data_file)

    for txt_type in ["src", "gen", "trg"]:
        txts = [d[txt_type] for d in data.values() if len(d['gen']) > 0]
        if append_tags:
            if txt_type in ["src", "trg"]:
                txts = [f"<t> {txt} </t>" for txt in txts]
            else:
                txts = [[f"<t> {hyp[0]} </t>"] for hyps in txts for hyp in hyps]

        if txt_type == "gen":
            txts = [t[0] for t in txts]

        out_file = f"/private/home/wangalexc/projects/qags/data/{txt_type}.txt"
        write_txt(txts, out_file)
        print(f"Wrote {len(txts)} texts to {out_file}")


def aggregate_questions():
    """ Extract questions generated from src, trg, and gen
    with the corresponding field from fseq logs (one log/txt) and write to jsonl.
    Each fseq log should have the txt field as 'source' (S)
    and the questions as generated 'hypotheses' (H) """

    for model in ["pgc-subset", "fan-subset", "bus-subset"]:
        for n_qsts in [10]:

            #src_qst_file = f"/checkpoint/wangalexc/fairseq/07-11-2019/qst.src-onmt-order.cnndm.test.out"
            #trg_qst_file = f"/checkpoint/wangalexc/fairseq/07-11-2019/qst.trg-onmt-order.cnndm.test.out"
            #gen_qst_file = f"/checkpoint/wangalexc/fairseq/07-18-2019/qst.{model}.cnndm.test.out"

            src_qst_file = f"/checkpoint/wangalexc/fairseq/08-14-2019/qst10.src-subset.cnndm.test.diverse.txt"
            gen_qst_file = f"/checkpoint/wangalexc/fairseq/08-14-2019/qst10.{model}.cnndm.test.diverse.txt"

            qst_files = {
                         "src": src_qst_file,
                         #"trg": trg_qst_file,
                         "gen": gen_qst_file
                        }

            all_txts, all_qsts = {}, {}
            for txt, qst_file in qst_files.items():
                txt_data = parse_generation(qst_file)
                all_txts[txt] = {k: v["src"] for k, v in txt_data.items()} # always grab "src"
                all_qsts[txt] = {k: v["gen"] for k, v in txt_data.items()} # always grab "src"


            # for each (question from a source x txt source) pair,
            # build the data then write out a SQuAD format file
            for txt_fld, qst_src in itertools.product(all_txts, all_qsts):
                txts = all_txts[txt_fld]
                qsts = all_qsts[qst_src]

                raw_data = {}
                assert txts.keys() == qsts.keys()
                sorted_keys = list(txts.keys())
                sorted_keys.sort()
                for k in sorted_keys:
                    txt = txts[k]
                    qst = qsts[k][:n_qsts]
                    raw_data[k] = {txt_fld: txt, "hypotheses": qst}

                data = format_squad(raw_data, context=txt_fld)
                out_dir = f"/private/home/wangalexc/projects/qags/data/labeled-subset/{model}"
                out_file = f"{out_dir}/qst{n_qsts}-{qst_src}-diverse.cnndm-{txt_fld}.json"
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                json.dump(data, open(out_file, "w", encoding="utf-8"))


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


def format_abstractive_qa():
    """ Format an extractive QA task as a freeform QA task """

    def _combine(x, y):
        """ Combine two texts, x and y """
        return f"{x} {y}"

    def _format_qa_split(data):
        """ """
        srcs, trgs = [], []

        for doc in data:
            for psg_d in doc["paragraphs"]:
                psg = psg_d["context"]
                for qst_d in psg_d["qas"]:
                    qst = qst_d["question"]
                    src = _combine(qst, psg)
                    srcs.append({"input": src})

                    anss = [a["text"] for a in qst_d["answers"]]
                    if len(anss) == 0: # question has no answer
                        ans = "<na>"
                    else:
                        ans = anss[0]
                    trgs.append({"target": ans})

        return srcs, trgs

    data_dir = "/private/home/wangalexc/data/squad/v2.0/original"
    out_dir = "/private/home/wangalexc/data/squad/v2.0/abstractive"
    split2file = {"train": "train.json",
                  "dev": "dev.json"
                  #"test": "test.json"
                 }

    for split_name, split_file in split2file.items():
        data = load_json(os.path.join(data_dir, split_file))["data"]
        srcs, trgs = _format_qa_split(data)
        write_data(srcs=srcs, trgs=trgs, out_dir=out_dir, out_prefix=split_name, out_format="jsonl")


def process_human_subset():

    label_map = {'Incorrect': 0, 'Unclear': 0, 'Correct': 1}
    txt_flds = ['src', 'trg', 'pgc', 'fan', 'bus']
    data_file = "data/aligned-subsets.json"
    data = json.load(open(data_file))

    for txt_fld in txt_flds:
        vals = list(data.values())
        txts = [v[txt_fld] for v in vals]
        if txt_fld in ['src', 'trg']:
            txts = [" ".join(ex) for ex in txts]
        else:
            txts = [" ".join([s['text'] for s in ex['sents'].values()]) for ex in txts]
            binary_scores = [label_map[v[txt_fld]['label']] for v in vals]
            count_scores = [sum([label_map[s['label']] for s in ex[txt_fld]['sents'].values()]) / len(ex[txt_fld]['sents']) for ex in vals]
        with open(f"data/labeled-subset/{txt_fld}-subset.txt", "w") as out_fh:
            for txt in txts:
                out_fh.write(f'{txt}\n')
        if txt_fld not in ['src', 'trg']:
            with open(f"data/labeled-subset/{txt_fld}-subset.scores.txt", "w") as out_fh:
                out_fh.write("\n".join(map(str, count_scores)))
            with open(f"data/labeled-subset/{txt_fld}-subset.binary-scores.txt", "w") as out_fh:
                out_fh.write("\n".join(map(str, binary_scores)))


#extract_src_trg_gen_from_fseq_log()
#extract_questions_and_write_jsonl()
aggregate_questions()
#format_abstractive_qa()
#process_human_subset()
