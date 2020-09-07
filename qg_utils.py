"""  """

import os
import re
import ast
import time
import json
import copy
import random
import argparse
import itertools
from tqdm import tqdm
from datetime import datetime
from functools import lru_cache
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize
from nltk import agreement
try:
    from nlgeval import compute_metrics, NLGEval
except ModuleNotFoundError as e:
    print("Unable to import NLGEval library!")
try:
    from bert_score import score as bert_score
except ModuleNotFoundError as e:
    print("Unable to import BERT Score!")
try:
    import krippendorff
except ModuleNotFoundError as e:
    print("Unable to import Krippendorff!")
from scipy.stats import pearsonr, spearmanr
import rouge

from utils import write_data, write_jsonl, write_txt, \
                  process, print_samples, format_squad, \
                  filter_line_fseq, parse_generation, \
                  load_txt, load_json
from eval_ppb_answers import evaluate, load_data, align_ans, count_noans


ANS_TOK = "[ANS]"
NO_ANS_TOK = "[NO_ANS]"


def extract_ans(txts):
    """ extract entities from a sentence using spacy

    rules:
        - entities (non-pronoun)
            - each portion of a person's name
        - noun chunks (non-pronoun)
            - adjectives within noun chunks
            - nouns w/ dependencies that are proper nouns, roughly nouns modifying proper nouns
            - if the head of a noun chunk if a verb, the entire noun chunk ?
        - for each conjunction,
            - the subtree of the head
            - the subtree of the children
    """
    nlp = get_spacy_nlp("en_core_web_lg")
    all_ans = list()
    for doc in nlp.pipe(txts, disable=[]):
        ans = list()
        for ent in doc.ents:
            ans.append(ent.text)
            #if not (len(ent) == 1 and ent[0].pos_ in ['PRON']):
            #    ans.append(ent.text)
            #if ent.label_ in ['PERSON']:
            #    for tok in ent:
            #        ans.append(tok.text)
        for chunk in doc.noun_chunks:
            ans.append(chunk.text)
            #if not (len(chunk) == 2 and chunk[0].pos_ in ['PRON']):
            #    ans.append(chunk.text)
            #for tok in chunk:
            #    if tok.pos_ in ['ADJ']:
            #        ans.append(tok.text)

            #    if tok.pos_ in ['NOUN'] and tok.head.pos_ in ['PROPN']:
            #        ans.append(tok.text)

            #    if tok.head.pos_ in ['VERB']:
            #        ans.append(' '.join([t.text for t in tok.head.subtree]))

        #specials = [t for t in doc if t.pos_ in ['SCONJ'] or t.tag_ in ['IN']]
        #for special in specials:
        #    ans.append(' '.join([t.text for t in special.head.subtree]))
        #    # subtrees of conjunctions
        #    for child in special.children:
        #        if child.is_punct or child.is_quote:
        #            continue
        #        ans.append(' '.join([t.text for t in child.subtree]))

        ans = list(set(ans))
        #ans = sorted(ans, key=lambda x: len(x))
        #ipdb.set_trace()
        all_ans.append(ans)
    return all_ans


def prepare_ans_conditional_data(data_file,
                                 out_dir,
                                 out_prefix,
                                 n_ans_per_txt=10,
                                 use_no_ans=False,
                                 use_only_no_ans=False,
                                 ):
    """ Given a text file, extract possible answer candidates for each line.

    Will generate n_ans_per_text instances for each line in txt
    """


    txt_w_ans_file = f"{out_dir}/{out_prefix}_w_{n_ans_per_txt}ans.txt"
    txt_file = f"{out_dir}/{out_prefix}.txt"
    ans_file = f"{out_dir}/{out_prefix}_{n_ans_per_txt}ans.txt"

    print(f"Preparing answer conditional question generation data for {data_file}")
    if use_only_no_ans:
        print("\twith ONLY NO_ANS!")
    elif use_no_ans:
        print("\twith NO_ANS option!")
    else:
        print("\twithout NO_ANS option!")

    all_txts = load_txt(data_file)
    print("Extracting entities...")
    all_anss = extract_ans(all_txts)
    print("\tDone!")
    print(f"\tMin ans count: {min(len(a) for a in all_anss)}")
    print(f"\tMax ans count: {max(len(a) for a in all_anss)}")

    print("Writing...")
    txts_w_ans = list()
    all_txt = list()
    all_ans = list()
    for txt, anss in zip(all_txts, all_anss):
        if use_only_no_ans:
            anss = [NO_ANS_TOK] * n_ans_per_txt
        elif use_no_ans:
            if len(anss) > n_ans_per_txt - 1:
                anss = random.sample(anss, k=n_ans_per_txt - 1)
            anss += [NO_ANS_TOK] * (n_ans_per_txt - len(anss))
            assert NO_ANS_TOK in anss, ipdb.set_trace()
        else:
            if len(anss) < n_ans_per_txt:
                extra_anss = random.choices(anss, k=n_ans_per_txt - len(anss))
                anss += extra_anss
            if len(anss) > n_ans_per_txt:
                anss = random.sample(anss, n_ans_per_txt)
            assert len(anss) == n_ans_per_txt, ipdb.set_trace()

        for ans in anss:
            txts_w_ans.append(f"{txt} {ANS_TOK} {ans}")
            all_txt.append(txt)
            all_ans.append(ans)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(txt_w_ans_file, 'w') as out_fh:
        for txt in txts_w_ans:
            out_fh.write(f'{txt}\n')
    with open(txt_file, 'w') as out_fh:
        for txt in all_txt:
            out_fh.write(f'{txt}\n')
    with open(ans_file, 'w') as out_fh:
        for ans in all_ans:
            out_fh.write(f'{ans}\n')
    print("\tDone!")
    print(f"\tWrote {len(txts_w_ans)} sentences to {txt_w_ans_file}")


def extract_gen_from_fseq_log(data_file, out_dir):
    """ """
    """ Extract source ('S'), target ('T'), and hypothesis generations ('H')
    from fseq logs and write each as a text file, one text per line.
    """
    data_file = "/checkpoint/wangalexc/fairseq/08-11-2019/qst.src-subset.cnndm.test.txt"

    append_tags = False
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

        #out_file = f"/private/home/wangalexc/projects/qags/data/{txt_type}.txt"
        out_file = f"{out_dir}/{txt_type}.txt"
        write_txt(txts, out_file)
        print(f"Wrote {len(txts)} texts to {out_file}")


def filter_qsts(qsts, n_qsts,
                prbs=None, reverse_prob=False,
                exp_anss=None, act_anss=None):
    """ Filter out questions by a number of criteria
    - repetitions: exact repetitions
    - length: short sentences are excluded

    If anss is nonempty, then this function expects that
        len(qsts) % len(ans) == 0 and that the questions
        are grouped by the answer.

    args:
        - qsts: questions
        - n_qsts: number of questions
        - prbs: probability of each question (optional, but not really)
        - reverse_prob: if True, sort by reverse probability
        - exp_anss: expected answers, e.g. that we conditioned on (optional)
        - act_anss: actual answers, e.g. from a QA model

    """

    qsts_and_prbs = zip(qsts, prbs)
    if act_anss is not None:
        qsts_and_prbs = [(q, p) for q, p , a in zip(qsts, prbs, act_anss) if a]
        n_qsts_w_ans = len(qsts_and_prbs)
    else:
        n_qsts_w_ans = None

    if act_anss is not None and exp_anss is not None:
        qsts_and_prbs = [(q, p) for q, p, a, e in zip(qsts, prbs, act_anss, exp_anss) if a == e]
        n_qsts_w_match_ans = len(qsts_and_prbs)
    else:
        n_qsts_w_match_ans = None
    qsts_and_prbs = sorted(qsts_and_prbs, key=lambda x: x[1], reverse=not reverse_prob)
    clean_qsts = list()
    clean_prbs = list()
    for qst, prob in qsts_and_prbs:
        try:
            qst_idx = qst.index('?') # get idx of *first* '?'
            # filter out stuff after '?'
            clean_qst = qst[:qst_idx + 1]
            clean_toks = clean_qst.split()
            if clean_qst in clean_qsts or len(clean_toks) < 3:
                continue
            clean_qsts.append(clean_qst)
            clean_prbs.append(prob)
        except ValueError as e: # no '?' mark
            continue

    n_clean_qsts = len(clean_qsts)
    if n_clean_qsts < n_qsts:
        #print("Too few questions!")
        supp_qsts = random.sample(qsts, n_qsts - n_clean_qsts)
        clean_qsts += supp_qsts

    ret = {
           'qsts': clean_qsts[:n_qsts],
           'n_qsts_w_match_ans': n_qsts_w_match_ans,
           'n_qsts_w_ans': n_qsts_w_ans,
           'n_clean_qsts': n_clean_qsts,
          }
    return ret


def main(arguments):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--command", choices=["extract_ans", "extract_gen", "filter_qsts"], description="Function to perform")
    parser.add_argument("--data_file", type=str, description="File from which to extract answers or filter questions. For `extract_ans`, this should be a text file with an example per line.")
    parser.add_argument("--out_dir", type=str, description="Directory to write outputs")
    parser.add_argument("--out_prefix", type=str, default="test", description="Prefix for files written out")

    # answer extraction options
    parser.add_argument("--n_ans", type=int, default=10, description="Number of answer candidates per example")

    args = parser.parse_args(arguments)

    if args.command == "extract_ans":
        prepare_ans_conditional_data(args.data_file, args.out_dir, args.out_prefix,
                                     n_ans_per_txt=args.n_ans)
    elif args.command == "extract_gen":
        extract_gen_from_fseq_log()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
