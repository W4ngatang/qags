""" Extract generations from fairseq outputs """

import os
import re
import json
import copy
import random
import itertools
from collections import defaultdict
from utils import filter_line_fseq, write_jsonl, write_text, \
                  process, print_samples, parse_generation, \
                  format_squad


N_SAMPLES = 5


def extract_src_trg_gen_from_fseq_log():
    """ Extract source ('S'), target ('T'), and hypothesis generations ('H')
    from fseq logs and write each as a text file, one text per line. """

    append_tags = False
    data_file = "/checkpoint/wangalexc/fairseq/07-01-2019/cnndm-summaries.out"
    #data_file = "/checkpoint/wangalexc/fairseq/07-11-2019/qst.gen.cnndm.test.out"
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
        write_text(txts, out_file)
        print(f"Wrote {len(txts)} texts to {out_file}")


def aggregate_questions():
    """ Extract questions generated from src, trg, and gen
    with the corresponding field from fseq logs (one log/txt) and write to jsonl.
    Each fseq log should have the txt field as 'source' (S)
    and the questions as generated 'hypotheses' (H) """

    #src_qst_file = "/checkpoint/wangalexc/fairseq/07-11-2019/qst.src.cnndm.test.out"
    #trg_qst_file = "/checkpoint/wangalexc/fairseq/07-11-2019/qst.trg.cnndm.test.out"
    #gen_qst_file = "/checkpoint/wangalexc/fairseq/07-11-2019/qst.gen.cnndm.test.out"

    model = "lstmsmalltied-beam10"
    date = "07-18-2018"
    src_qst_file = f"/checkpoint/wangalexc/fairseq/07-11-2019/qst.src-onmt-order.cnndm.test.out"
    trg_qst_file = f"/checkpoint/wangalexc/fairseq/07-11-2019/qst.trg-onmt-order.cnndm.test.out"
    gen_qst_file = f"/checkpoint/wangalexc/fairseq/07-18-2019/qst.{model}.cnndm.test.out"

    qst_files = {
                 "src": src_qst_file,
                 "trg": trg_qst_file,
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
            qst = qsts[k]
            raw_data[k] = {txt_fld: txt, "hypotheses": qst}

        data = format_squad(raw_data, context=txt_fld)
        out_dir = f"/private/home/wangalexc/projects/qags/data/{model}"
        out_file = f"{out_dir}/qst-{qst_src}.cnndm-{txt_fld}.json"
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

#extract_src_trg_gen_from_fseq_log()
#extract_questions_and_write_jsonl()
aggregate_questions()
