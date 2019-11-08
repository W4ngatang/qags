""" Extract generations from fairseq outputs """

import os
import re
import ast
import time
import json
import copy
import random
import itertools
from datetime import datetime
from collections import defaultdict, Counter

import ipdb
import rouge
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from nltk.tokenize import sent_tokenize
from nltk import agreement
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler

from utils import write_data, write_jsonl, write_txt, \
                  process, print_samples, format_squad, \
                  filter_line_fseq, parse_generation, \
                  load_txt, load_json
from eval_ppb_answers import evaluate, load_data, align_ans

N_SAMPLES = 5
ATTN_IDXS = [-2, -3]
MTURK_BAD_RESPONSES = ['[DISCONNECT]', '[RETURNED]']


def get_qags_scores(src_ans_file, trg_ans_file, metric_name="em", n_qsts_per_doc=10):
    """Load answer files and compute similarity scores
    """
    srcs = load_data(src_ans_file)
    trgs = load_data(trg_ans_file)
    src_ans, trg_ans = align_ans(srcs, trgs)
    qags_scores, _, _, _, _ = evaluate(tgts=src_ans, prds=trg_ans,
                                       n_qsts_per_doc=n_qsts_per_doc,
                                       metric_name=metric_name)
    return qags_scores


def get_rouge_scores(hyps, refs):
    """ Get ROUGE scores between hyps and refs.
    Computes ROUGE-{1,2,3,4,L} and averages F1 for each.
    """
    rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                             max_n=4,
                             limit_length=True,
                             length_limit=100,
                             length_limit_type='words',
                             apply_avg=False,
                             apply_best=False,
                             alpha=0.5,
                             weight_factor=1.2,
                             stemming=True)
    rouge_d = rouge_eval.get_scores(hyps, refs)
    rouge_scores = [[] for _ in range(len(hyps))]
    for metric_name, metric_d in rouge_d.items():
        for ex_idx, ex_metrics in enumerate(metric_d):
            rouge_scores[ex_idx].append(ex_metrics['f'][0])
    rouge_scores = [sum(l) / len(rouge_d) for l in rouge_scores]
    return rouge_scores


def get_lens(txts):
    """ """
    return [len(txt.split()) for txt in txts]

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


def extract_subset():
    """ Given a list of aligned files, extract a (random) subset """

    n_exs = 5
    min_len = 200
    max_len = 400
    curr_time = datetime.now()
    sp2files = {
            "src": ("data/xsum.test.src.all.txt", f"data/xsum.test.src.{curr_time.strftime('%m%d%H%M')}.random{n_exs}.txt"),
            "trg": ("data/xsum.test.trg.all.txt", f"data/xsum.test.trg.{curr_time.strftime('%m%d%H%M')}.random{n_exs}.txt"),
            "bart": ("data/xsum.test.bart.all.txt", f"data/xsum.test.bart.{curr_time.strftime('%m%d%H%M')}.random{n_exs}.txt"),
             }

    lens = {
            "src": np.array(get_lens(load_txt(sp2files["src"][0]))),
            "trg": np.array(get_lens(load_txt(sp2files["trg"][0]))),
            "bart": np.array(get_lens(load_txt(sp2files["bart"][0]))),
           }

    # count # exs and get random subset
    srcs = load_txt(sp2files["src"][0])
    all_idxs = [i for i, s in enumerate(srcs) if len(s.split()) <= max_len and len(s.split()) >= min_len]
    idxs = random.sample(all_idxs, n_exs)
    print(f"\tSampled {n_exs} examples from {len(all_idxs)} considered")

    for in_file, out_file in sp2files.values():
        with open(in_file, encoding="utf-8") as in_fh:
            all_data = in_fh.readlines()
        out_data = [all_data[i] for i in idxs]
        if "src" in in_file:
            out_lens = np.array(get_lens(out_data))
            print(f"Mean src len: {np.mean(out_lens)}")
            print(f"Median src len: {np.median(out_lens)}")
            print(f"Max src len: {np.max(out_lens)}")
            print(f"Min src len: {np.min(out_lens)}")
        with open(out_file, 'w', encoding="utf-8") as out_fh:
            for out_datum in out_data:
                out_fh.write(f"{out_datum}")

    print(f"Done!")


def aggregate_questions():
    """ Extract questions generated from src, trg, and gen
    with the corresponding field from fseq logs (one log/txt) and write to jsonl.
    Each fseq log should have the txt field as 'source' (S)
    and the questions as generated 'hypotheses' (H) """

    #for model in ["pgc-subset", "fan-subset", "bus-subset"]:
    for ckpt in ["best"]:
        for n_qsts in [5]:
            model = "bus-subset"

            #src_qst_file = f"/checkpoint/wangalexc/fairseq/07-11-2019/qst.src-onmt-order.cnndm.test.out"
            #trg_qst_file = f"/checkpoint/wangalexc/fairseq/07-11-2019/qst.trg-onmt-order.cnndm.test.out"
            #gen_qst_file = f"/checkpoint/wangalexc/fairseq/07-18-2019/qst.{model}.cnndm.test.out"

            src_qst_file = f"/checkpoint/wangalexc/fairseq/10-11-2019/qst5-ckpt{ckpt}.src-subset.cnndm.test.txt"
            gen_qst_file = f"/checkpoint/wangalexc/fairseq/11-11-2019/qst5-ckpt{ckpt}.{model}.cnndm.test.txt"

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
                out_file = f"{out_dir}/qst{n_qsts}-ckpt{ckpt}-{qst_src}.cnndm-{txt_fld}.json"
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                json.dump(data, open(out_file, "w", encoding="utf-8"))


JOIN_PUNCT = {'- lrb -': '-lrb-', '- rrb -': '-rrb-',
              'n \' t ': 'n\'t ', '\' s ': '\'s ', ' \' ve ': '\'ve ',
              ' \' m ': '\'m ', ' \' re ': ' \'re ', '\' d ': '\'d ',
              ' \' ll ': ' \'ll '
             }
REPLACE_PUNCT = {'-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']', '#': '$'}
NO_LEADING_SPACE_PUNCT = ['.', ',', '\'s', '\'m', '\'ve', '\'d', '?', '!', 'n\'t', '\'re', '\'']
NO_TRAILING_SPACE_PUNCT = [' `']#, ' \'']


def detokenize_sent(sent):
    """ Detokenize sents for readability, including:

    - capitalizing first word of sentence (missing for proper nouns for English)
    - remove extra spaces
    - swap -lrb- and -rrb- for [, ] respectively
    """
    sent = sent.capitalize()
    for k, v in REPLACE_PUNCT.items():
        sent = sent.replace(k, v)
    for punct in NO_LEADING_SPACE_PUNCT:
        sent = sent.replace(f' {punct}', punct)
    for punct in NO_TRAILING_SPACE_PUNCT:
        sent = sent.replace(f'{punct} ', punct)
    return sent


def align_summaries():
    """Align summaries that may be in different order

    Strategy: represent as bag of words / ngram / char
        and find nearest neighbors.
    """

    def get_aligned_shortest(txts1, txts2, n_exs_to_search):
        """Get alignment in txt2 of the shortest n_exs summaries in txts1
        """
        lens1 = [len(t.split()) for t in txts1]
        cnts1 = [Counter(t.split()) for t in txts1]
        sorted_txts2 = sorted(enumerate(txts2), key=lambda x: len(x[1].split()))
        #cnts2 = [(i, Counter(t.split())) for i, t in sorted_txts2[:n_exs_to_search]]

        #diffs = [[(i2, sum((c1 & c2).values())) for i2, c2 in cnts2] for c1 in cnts1]
        idxs2 = list()
        for len1, cnt1 in zip(lens1, cnts1):
        #for diff in diffs:
            cnts2 = [(i, Counter(t.split()[:len1])) for i, t in sorted_txts2[:n_exs_to_search]]
            diff = [(i2, sum((cnt1 & c2).values())) for i2, c2 in cnts2]
            new_idx2, (orig_idx2, cnt2) = max(enumerate(diff), key=lambda x: x[1][-1])
            #idx2, cnt2 = max(diff, key=lambda x: x[-1])
            idxs2.append(orig_idx2)
            cnts2.pop(new_idx2)

        return idxs2

    def get_aligned(txts1, txts2, idxs, search_width):
        """Get alignment in txt2 in txts1
        """
        #sorted_txts2 = sorted(enumerate(txts2), key=lambda x: len(x[1].split()))
        sorted_txts2 = sorted(enumerate(txts2), key=lambda x: x[1])
        cnts1 = [Counter(t.split()) for t in txts1]
        idxs2 = list()
        for idx, cnt in zip(idxs, cnts1):
            start_idx = max(0, idx - search_width)
            end_idx = min(len(txts2), idx + search_width)
            cnts2 = [(i, Counter(t.split())) for i, t in sorted_txts2[start_idx: end_idx]]
            diffs = [(i2, sum((cnt & c2).values())) for i2, c2 in cnts2]
            idx2, _ = max(diffs, key=lambda x: x[-1])
            idxs2.append(idx2)
        return idxs2

    def proc(txts, proc_d):
        """ """
        new_txts = list()
        for txt in txts:
            for k, v in proc_d.items():
                txt = txt.replace(k ,v)
            new_txts.append(txt)
        return new_txts

    n_exs = 100
    search_width = 100
    shortest = 0 # if 0, random

    mdl_files = {
                 "bus": ("data/all_bus/all_bus.trg.txt", "data/all_bus/all_bus.src.400words.txt",
                     {'-': ' - ', '`': ' ` ', '\'': ' \' ', '.': ' . ', '': ''}),
                 "fas": ("data/all_fas/all_fas_rerank.trg.v2.txt", "/misc/vlgscratch4/BowmanGroup/awang/raw_data/cnndm_harvard/test.txt.src",
                         {'-': ' - ', '`': ' ` ', '\'': ' \' ', '.': ' . '}),
                 "pgc": ("data/all_pgc/all_pgc.trg.txt", "data/all_pgc/all_pgc.src.txt",
                         {'(': '- lrb -', ')': '- rrb -', '`': ' ` ', '\'': ' \' ', ',': ' , '}),
                }

    ref_src_file = "/misc/vlgscratch4/BowmanGroup/awang/processed_data/cnndailymail/fseq/src.cnndm.test.txt"
    ref_trg_file = "/misc/vlgscratch4/BowmanGroup/awang/processed_data/cnndailymail/fseq/trg.cnndm.test.txt"
    # Load the sources that we'll use
    orig_preproc_d = {} #{'- lrb -': '-lrb-', '- rrb -': '-rrb-'}
    all_ref_srcs = proc(load_txt(ref_src_file), orig_preproc_d)
    all_ref_trgs = proc(load_txt(ref_trg_file), orig_preproc_d)
    #all_sorted_ref_srcs_and_idxs = sorted(enumerate(all_ref_srcs), key=lambda x: len(x[1].split()))
    all_sorted_ref_srcs_and_idxs = sorted(enumerate(all_ref_srcs), key=lambda x: x[1])
    all_sorted_ref_idxs, all_sorted_ref_srcs = zip(*all_sorted_ref_srcs_and_idxs)
    all_sorted_ref_trgs = [all_ref_trgs[i] for i in all_sorted_ref_idxs]
    all_sorted_ref_lens = [len(s.split()) for s in all_sorted_ref_srcs]
    min_ref_src_len = min(all_sorted_ref_lens)

    if shortest:
        ref_srcs = all_sorted_ref_srcs[:n_exs]
        ref_trgs = all_sorted_ref_trgs[:n_exs]
        write_txt(ref_srcs, f"data/subset{n_exs}.shortest.src.txt")
        write_txt(ref_trgs, f"data/subset{n_exs}.shortest.trg.txt")
    else:
        #rand_idxs = random.sample(range(int(len(all_sorted_ref_srcs) / 2)), n_exs)
        rand_idxs = random.sample(range(len(all_sorted_ref_srcs)), n_exs)
        ref_srcs = [all_sorted_ref_srcs[i] for i in rand_idxs]
        ref_trgs = [all_sorted_ref_trgs[i] for i in rand_idxs]
        write_txt(ref_srcs, f"data/subset{n_exs}.random.src.txt")
        write_txt(ref_trgs, f"data/subset{n_exs}.random.trg.txt")
        print(f"idxs: {rand_idxs}")
    ref_lens = np.array([len(s.split()) for s in ref_srcs])

    for mdl_name, (mdl_gen_file, mdl_ref_file, mdl_preproc_d) in mdl_files.items():
        print(f"Processing data for {mdl_name}")
        mdl_gens = load_txt(mdl_gen_file)
        mdl_refs = proc(load_txt(mdl_ref_file), mdl_preproc_d)
        print("\tFinished loading data")

        # maps from ref order idx to mdl order idx
        start_time = time.time()
        if "src" in mdl_ref_file: # compare against sources
            if shortest:
                mdl_idxs = get_aligned_shortest(ref_srcs, mdl_refs,
                                                n_exs_to_search=n_exs + search_width)
            else:
                mdl_idxs = get_aligned(ref_srcs, mdl_refs, rand_idxs, search_width)
        else: # compare against gold targets / references
            assert "ref" in mdl_ref_file
            if shortest:
                mdl_idxs = get_aligned_shortest(ref_trgs, mdl_refs, n_exs_to_search=n_exs + search_width)
            else:
                mdl_idxs = get_aligned(ref_trgs, mdl_refs, rand_idxs, search_width)
        subset_mdl_gens = [mdl_gens[i] for i in mdl_idxs]
        subset_mdl_lens = np.array([len(s.split()) for s in subset_mdl_gens])
        #print(f"mdl idxs: {mdl_idxs}")
        print(f"\tFinished aligning {n_exs} examples in {time.time() - start_time}s")
        print(f"\tMean src length: {np.mean(ref_lens)}")
        print(f"\tMax src length: {np.max(ref_lens)}")
        print(f"\tMean gen length: {np.mean(subset_mdl_lens)}")
        print(f"\tMax gen length: {np.max(subset_mdl_lens)}")

        if shortest:
            write_txt(subset_mdl_gens, f"data/subset{n_exs}.{mdl_name}.shortest.ref_order.txt")
        else:
            write_txt(subset_mdl_gens, f"data/subset{n_exs}.{mdl_name}.random.ref_order.txt")
        print("\tFinished writing data")

    return


def prepare_parlai_data():
    """ Prepare data for ParlAI mturk tasks """

    # load original articles
    mdl_files = {

                # labelled subset from Falke et al., 2019
                (100, 'subset'): {
                     'src': 'data/subset-src.txt',
                     'bus': 'data/subset-bus.txt',
                     'fas': 'data/subset-fas.txt',
                     'pgc': 'data/subset-pgc.txt',
                     'trg': 'data/subset-trg.txt'
                 },

                 # 500 shortest examples
                 (500, 'shortest'): {
                     'src': 'data/subset500.src.txt',
                     'trg': 'data/subset500.trg.txt',
                     'bus': 'data/subset500.bus.trg.ref_order.txt',
                     'fas': 'data/subset500.fas.trg.ref_order.txt',
                     'pgc': 'data/subset500.pgc.trg.ref_order.txt',
                 },

                 # random 1000 examples
                 (1000, 'random'): {
                     'src': 'data/subset1000.random.src.txt',
                     'trg': 'data/subset1000.random.trg.txt',
                     'bus': 'data/subset1000.bus.random.ref_order.txt',
                     'fas': 'data/subset1000.fas.random.ref_order.txt',
                     'pgc': 'data/subset1000.pgc.random.ref_order.txt',
                 },

                 # random 100 examples
                 (100, 'random'): {
                     'src': 'data/subset100.random.src.txt',
                     'trg': 'data/subset100.random.trg.txt',
                     'bus': 'data/subset100.bus.random.ref_order.txt',
                     'fas': 'data/subset100.fas.random.ref_order.txt',
                     'pgc': 'data/subset100.pgc.random.ref_order.txt',
                 },

                 (5, 'xsum-random'): {
                     'src': 'data/xsum.test.src.10301218.random5.txt',
                     'trg': 'data/xsum.test.trg.10301218.random5.txt',
                     'bart': 'data/xsum.test.bart.10301218.random5.txt',
                 },

                 (10, 'xsum-random'): {
                     'src': 'data/xsum.test.src.10231603.random10.txt',
                     'trg': 'data/xsum.test.trg.10231603.random10.txt',
                     'bart': 'data/xsum.test.bart.10231603.random10.txt',
                 },

                 (100, 'xsum-random'): {
                     'src': 'data/xsum.test.src.10251018.random100.txt',
                     'trg': 'data/xsum.test.trg.10251018.random100.txt',
                     'bart': 'data/xsum.test.bart.10251018.random100.txt',

                 },

                 (1000, 'xsum-random'): {
                     'src': 'data/xsum.test.src.10251125.random1000.txt',
                     'trg': 'data/xsum.test.trg.10251125.random1000.txt',
                     'bart': 'data/xsum.test.bart.10251125.random1000.txt',
                 },
                }

    n_shards = 20
    n_exs = 1000
    dataset = 'xsum'
    subset_key = (n_exs, f'{dataset}-random')
    should_add_attn_task = 1
    should_filter_length = 0
    mdl_files = mdl_files[subset_key]

    raw_srcs = [s.strip() for s in open(mdl_files['src'], encoding="utf-8")]
    srcs = []
    for src in raw_srcs:
        for k, v in JOIN_PUNCT.items():
            src = src.replace(k, v)
        srcs.append(src)
    if dataset == 'xsum':
        srcs_sents = [[detokenize_sent(s) for s in sent_tokenize(src)] for src in srcs]
    elif dataset == 'cnndm':
        srcs_sents = [[detokenize_sent(s) for s in sent_tokenize(src)] for src in srcs]

    if should_filter_length:
        #lens = np.array([len(s.split()) for s in srcs])
        lens = np.array([len(s) for s in srcs])
        idxs = lens.argsort().tolist()
    else:
        idxs = list(range(len(srcs)))
    assert isinstance(idxs, list), "Idxs is wrong type"

    # for each model
    for mdl_name, mdl_file in mdl_files.items():
        # load the generations
        raw_gens = [g.strip() for g in open(mdl_file)]
        gens = []
        for gen in raw_gens:
            for k, v in JOIN_PUNCT.items():
                gen = gen.replace(k, v)
            gens.append(gen)

        para_data = []
        sent_data = []

        for src_idx in idxs:
            src = srcs[src_idx]
            src_sents = srcs_sents[src_idx]
            gen = gens[src_idx]
            par_sents = []

            if dataset == 'xsum' and mdl_name != 'src':
                gen_sents = [detokenize_sent(gen)]
            else:
                gen_sents = [detokenize_sent(s) for s in sent_tokenize(gen)]
            for sent_idx, sent in enumerate(gen_sents):
                gen_d = {'dialog': [{'speaker': 'model', 'text': sent}],
                         'ex_idx': (mdl_name, src_idx, sent_idx),
                         'para_idx': sent_idx
                        }
                par_sents.append(gen_d)

            if should_add_attn_task:
                # negative attn task
                if dataset == 'xsum':
                    sents = random.sample(src_sents, 2)
                    words1 = sents[0].split()
                    words2 = sents[1].split()
                    sent = []
                    for i in range(max(len(words1), len(words2))):
                        if (((i // 2) % 2 == 0) and not (i >= len(words1))) or \
                            (i >= len(words2)):
                            sent.append(words1[i])
                        else:
                            sent.append(words2[i])
                    sent = ' '.join(sent).lower().capitalize()
                    #sent = ' '.join(words1[:2] + words2[2:]).lower().capitalize()
                else:
                    sent = random.choice(gen_sents).split()
                    random.shuffle(sent)
                    sent = ' '.join(sent).lower().capitalize()

                sent_idx = -2
                gen_d = {'dialog': [{'speaker': 'model', 'text': sent}],
                         'ex_idx': (mdl_name, src_idx, sent_idx),
                         'para_idx': sent_idx,
                         'answer': 'no'
                        }
                par_sents.append(gen_d)

                # positive attn task
                pos_idx = 1 if dataset == 'xsum' else 1
                sent = src_sents[min(len(src_sents), pos_idx)]
                sent_idx = -3
                gen_d = {'dialog': [{'speaker': 'model', 'text': sent}],
                         'ex_idx': (mdl_name, src_idx, sent_idx),
                         'para_idx': sent_idx,
                         'answer': 'yes'
                        }
                par_sents.append(gen_d)


            para_d = {
                      'dialog': [{'speaker': 'model', 'text': ' '.join(gen_sents)}],
                      'ex_idx': (mdl_name, src_idx, -1),
                      'para_idx': 0
                     }
            para_data.append(para_d)
            sent_data.append(par_sents)
        assert len(para_data) == len(sent_data), "Different number of paragraphs and sentences!"

        n_exs_per_shard = len(para_data) // n_shards
        for shard_n in range(n_shards):
            para_file = f"data/mturk/{dataset}/{mdl_name}_para_nex{n_exs}_randorder_shard{shard_n}.jsonl"
            with open(para_file, 'w', encoding='utf-8') as para_fh:
                data_slice = para_data[shard_n * n_exs_per_shard : (shard_n + 1) * n_exs_per_shard]
                for para_d in data_slice:
                    para_fh.write(f"{json.dumps(para_d)}\n")

            # write to jsonl
            sent_file = f"data/mturk/{dataset}/{mdl_name}_sent_nex{n_exs}_randorder_shard{shard_n}.jsonl"
            with open(sent_file, 'w', encoding='utf-8') as sent_fh:
                data_slice = sent_data[shard_n * n_exs_per_shard : (shard_n + 1) * n_exs_per_shard]
                for sents_d in data_slice:
                    for sent_d in sents_d:
                        sent_fh.write(f"{json.dumps(sent_d)}\n")



def compute_correlations_with_human(turk_files, ref_file, hyp_file, mdl,
                                    qags_src_file, qags_trg_file, n_qsts_per_doc):
    """ Compute sentence and system level correlations
    between human annotations and ROUGE scores
    """

    # 1 is YES, 2 is NO
    resp_map = {'1': 1, '2': 0}

    # Load mturk data
    n_hits, n_rejected_hits, n_total_hits = 0, 0, 0
    idxs = list()
    idx2responses = defaultdict(lambda: defaultdict(list))
    for turk_file in turk_files:
        mturk_data = [ast.literal_eval(l) for l in open(turk_file, encoding="utf-8")]
        for datum in mturk_data:
            for worker_id, worker in datum['worker_data'].items():

                n_total_hits += 1
                # Filter out bad reponses
                bad_resp_flag, short_msg_flag, attn_fail_flag = 0, 0, 0
                ## filter out returns and discounnects
                if worker['response']['text'] in MTURK_BAD_RESPONSES:
                    bad_resp_flag = 1
                ## filter out short responses
                if 'task_data' in worker['response']:
                    for response in worker['response']['task_data']:
                        if not response.get('textReason', ''):
                            short_msg_flag = True
                    ## filter out attn check fails
                    for task_idx, task in enumerate(worker['task_data']):
                        if task['conversations'][1].get('answer', None) is not None:
                            choice = int(worker['response']['task_data'][task_idx]['speakerChoice'])
                            expected = 1 if task['conversations'][1]['answer'] == 'yes' else 2
                            if choice != expected:
                                attn_fail_flag = True
                # filter out too short time
                if bad_resp_flag or short_msg_flag or attn_fail_flag:
                    n_rejected_hits += 1
                    continue

                n_hits += 1
                para_idx = tuple(worker['task_data'][0]['conversations'][0]['ex_idx'])[1]
                sent_idxs = [t['conversations'][1]['ex_idx'][2] for t in worker['task_data']]
                resps = [d["speakerChoice"] for d in worker['response']['task_data']]
                for sent_idx, resp in zip(sent_idxs, resps):
                    idx2responses[para_idx][sent_idx].append(resp_map[resp])
                idxs.append(para_idx)

    idxs = list(set(idxs))


    def print_response(par_idx):
        src_key = ('src', par_idx, -1)
        n_trgs = len(idx2responses[src_key])
        print(f"Paragraph {par_idx}")
        print(f"\t{idx2data[src_key]}")
        #for i in range(n_trgs):
        for trg_key in idx2responses[src_key].keys():
            #trg_key = (mdl, par_idx, i)
            print(f"\t{idx2responses[src_key][trg_key]}: {idx2data[trg_key]}")


    # Aggregate stuff
    n_tasks = 0 # n article-sentence pairs
    n_yes, n_no = 0, 0 # n aggregate yes/no among article-sentence pairs
    n_all_votes_yes, n_all_votes_no = 0, 0 # n tasks where all voted yes/no
    n_all_responses_yes, n_all_responses_no = 0, 0 # n articles where all tasks are yes/no
    human_scores = list() # scores per summary, averaged over sentences
    odd_human_scores = list() # scores for summaries w/ 3 annotations
    odd_idxs = list() # idxs of summaries w/ 3 annotations
    n_responses = defaultdict(int) # n tasks w/ {1,2,3} responses
    odd_kappas = defaultdict(lambda: defaultdict(list))
    for para_idx in idxs:
        para_d = idx2responses[para_idx]
        agg_labels = []
        odd_agg_labels = []
        n_par_tasks, n_par_yes = 0, 0
        for sent_idx, votes in para_d.items():
            if sent_idx in ATTN_IDXS:
                continue
            assert votes, "No votes!"
            #votes = votes[:3]
            votes0 = votes.count(0)
            votes1 = votes.count(1)
            if votes1 >= votes0:
                agg_labels.append(1)
                n_yes += 1
                n_par_yes += 1
            else:
                agg_labels.append(0)
                n_no += 1
            n_responses[votes0 + votes1] += 1

            # sentence level bookkeeping
            if votes1 == len(votes):
                n_all_votes_yes += 1
            if votes0 == len(votes):
                n_all_votes_no += 1
            if len(votes) % 2 == 1 and len(votes) > 1:
                odd_agg_labels.append(1 if votes1 > votes0 else 0)
                if para_idx not in odd_idxs:
                    odd_idxs.append(para_idx)
                odd_kappas[para_idx][sent_idx] = votes
            n_tasks += 1
            n_par_tasks += 1

        # article level bookkeeping
        human_scores.append(sum(agg_labels) / len(agg_labels))
        if odd_agg_labels:
            odd_human_scores.append(sum(odd_agg_labels) / len(odd_agg_labels))
        if n_par_yes == n_par_tasks: # attn task
            n_all_responses_yes += 1
        if n_par_yes == 0:
            n_all_responses_no += 1

    print(f"Loaded data from {len(idxs)} articles, {n_tasks} tasks, {n_hits} HITS")
    print(f"\tn rejected {n_rejected_hits}, n total HITS {n_total_hits}")
    print(f"\tn_yes responses {n_yes}; n_no responses {n_no}")
    print(f"\tn tasks all responses yes {n_all_votes_yes}; no {n_all_votes_no}; n_disagreement {n_tasks - n_all_votes_yes - n_all_votes_no}")
    print(f"\t{len(odd_human_scores)} / {len(human_scores)} ({100 * len(odd_human_scores)/len(human_scores):.2f}%) articles with odd number of labels")
    print(f"\t{n_all_responses_yes} / {len(idxs)} ({100 * n_all_responses_yes / len(idxs):.2f}%) articles where all tasks are yes")
    print(f"\t{n_all_responses_no} / {len(idxs)} ({100 * n_all_responses_no / len(idxs):.2f}%) articles where all tasks are no")
    #print(f"\t{', '.join([str(k) + ':' + str(v) for k, v in n_responses.items()])}")
    for k, v in n_responses.items():
        print(f"\t{v} tasks with {k} responses")
    print()


    def compute_fleiss(resps):
        """ Janky because we don't really have three annotators """
        M = []
        for par_idx, sents in resps.items():
            for sent_idx, r in sents.items():
                if sent_idx in ATTN_IDXS:
                    continue
                votes = [0] * 2
                votes[0] = r.count(0)
                votes[1] = r.count(1)
                M.append(votes)
        M = np.array(M)
        N, k = M.shape  # N is # of items, k is # of categories
        n_annotators = float(np.sum(M[0, :]))  # # of annotators
        p = np.sum(M, axis=0) / (N * n_annotators) # prob of being rated 0, 1
        P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
        Pbar = np.sum(P) / N
        PbarE = np.sum(p * p)

        kappa = (Pbar - PbarE) / (1 - PbarE)
        print(f"Fleiss: {kappa}, n annotators {n_annotators}")

    def compute_rouge_correlation(idxs, scores):
        """Compute ROUGE correlation with some scores
        """
        all_hyps = [l.strip() for l in open(hyp_file, encoding='utf-8')]
        all_refs = [l.strip() for l in open(ref_file, encoding='utf-8')]
        refs = [all_refs[idx] for idx in idxs]
        hyps = [all_hyps[idx] for idx in idxs]
        rouge_scores = get_rouge_scores(hyps, refs)
        pearson_corr = pearsonr(scores, rouge_scores)
        spearman_corr = spearmanr(scores, rouge_scores)
        print(f"pearson correlation w/ ROUGE: {pearson_corr}")
        print(f"spearman correlation w/ ROUGE: {spearman_corr}")

    def compute_qags_correlation(idxs, scores, metric_name):
        """Compute QAGS correlation with some scores
        """

        all_qags_scores = get_qags_scores(qags_src_file, qags_trg_file,
                                          metric_name=metric_name,
                                          n_qsts_per_doc=n_qsts_per_doc)
        qags_scores = [all_qags_scores[idx] for idx in idxs]
        pearson_corr = pearsonr(scores, qags_scores)
        spearman_corr = spearmanr(scores, qags_scores)
        print(f"pearson correlation w/ QAGS {metric_name}: {pearson_corr}")
        print(f"spearman correlation w/ QAGS {metric_name}: {spearman_corr}")


    print(f"All examples")
    compute_rouge_correlation(idxs, human_scores)
    if qags_src_file:
        compute_qags_correlation(idxs, human_scores, metric_name="em")
        compute_qags_correlation(idxs, human_scores, metric_name="f1")
    compute_fleiss(idx2responses)
    print()

    print(f"Examples with odd # labels ({len(odd_idxs)})")
    compute_rouge_correlation(odd_idxs, odd_human_scores)
    if qags_src_file:
        compute_qags_correlation(odd_idxs, odd_human_scores, metric_name="em")
        compute_qags_correlation(odd_idxs, odd_human_scores, metric_name="f1")
    compute_fleiss(odd_kappas)



def mturk_posthoc(is_sandbox=False):
    """Currently: analyze time
    """
    data_handler = MTurkDataHandler(file_name='pmt_sbdata.db' if is_sandbox else 'pmt_data.db')

    all_runs = data_handler.get_all_run_data()

    times = []
    statuses = []
    for run in all_runs[-10:]:
        asgs = data_handler.get_assignments_for_run(run['run_id'])
        for asg in asgs:
            asg_data = data_handler.get_worker_assignment_pairing(asg['worker_id'], asg['assignment_id'])
            if asg_data['status'] in ['disconnect']:
                continue
            if asg_data['task_end'] is None or asg_data['task_start'] is None:
                continue
            statuses.append(asg_data['status'])
            times.append(asg_data['task_end'] - asg_data['task_start'])

    times = np.array(times)
    print(f"Summary of reseponse times for {len(times)} assignments:")
    print(f"mean: {times.mean()} ")
    print(f"std: {times.std()}")
    print(f"min: {times.min()}")
    print(f"max: {times.max()}")
    ipdb.set_trace()




subset100_data = {
    "bus": ["data/mturk/summary/precision/mturk_data.09271534.jsonl",
            "data/mturk/summary/precision/mturk_data.10041456.jsonl",
            "data/mturk/summary/precision/mturk_data.10071418.jsonl"],

    "trg": ["data/mturk/summary/precision/mturk_data.09271635.jsonl"],

    "pgc": [
            "data/mturk/summary/precision/mturk_data.09271736.jsonl",
            "data/mturk/summary/precision/mturk_data.10021638.jsonl",
            "data/mturk/summary/precision/mturk_data.10031605.jsonl"
           ],

    "fas": ["data/mturk/summary/precision/mturk_data.10011138.jsonl",
            "data/mturk/summary/precision/mturk_data.10021758.jsonl",
            "data/mturk/summary/precision/mturk_data.10071607.jsonl",
           ],

    "hyp": {mdl: f"data/subset-{mdl}.txt" for mdl in ["bus", "pgc", "fas"]},
    "ref": "data/subset-trg.txt",
    #qags_src_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset/prd.qst{n_qsts_per_doc}-gen.cnndm-src.json"
    #qags_trg_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset/prd.qst{n_qsts_per_doc}-gen.cnndm-gen.json"

}

subset500_data = {
    "bus": [
            "data/mturk/summary/precision/mturk_data.10111337.jsonl",
            # NOTE(Alex): 10/14 10:59 didn't use HIT requirements
            #"data/mturk/summary/precision/mturk_data.10141059.jsonl",
            # NOTE(Alex): contains 2x annotations / task
            "data/mturk/summary/precision/mturk_data.10141206.jsonl",
           ],

    "pgc": [
           ],

    "fas": [
           ],

    "hyp": {mdl: f"data/subset500.{mdl}.trg.ref_order.txt" for mdl in ["bus", "pgc", "fas"]},
    "ref": "data/subset500.trg.txt",

}

subset1000_data = {
    "bus": [
            # order: shard 0, 1, 2, ...
            "data/mturk/summary/precision/mturk_data.10211029.jsonl",
            "data/mturk/summary/precision/mturk_data.10211306.jsonl",
            "data/mturk/summary/precision/mturk_data.10211417.jsonl",
            "data/mturk/summary/precision/mturk_data.10231357.jsonl",
            "data/mturk/summary/precision/mturk_data.10231529.jsonl",
            # RERUN THE REMAINDER (missing one or two annotations)
           ],

    "pgc": [
           ],

    "fas": [
           ],

    "hyp": {mdl: f"data/subset1000.{mdl}.random.ref_order.txt" for mdl in ["bus", "pgc", "fas"]},
    "ref": "data/subset1000.random.trg.txt",

}

xsum_subset1000_data = {
    "bart": [
            # order: shard 0, 1, 2, ...
            "data/mturk/xsum/precision/mturk_data.11051235.jsonl",
            "data/mturk/xsum/precision/mturk_data.11060913.jsonl",
           ],

    "hyp": {"bart": "data/xsum.test.bart.10251125.random1000.txt"},
    "ref": "data/xsum.test.trg.10251125.random1000.txt",
}

# Settings
mdl = "bart"
exp_name = "xsum-subset1000"

if exp_name == "subset500":
    exp_d = subset500_data
    n_qsts_per_doc = 5
    qags_src_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset500/prd.qst{n_qsts_per_doc}-ckptbest-gen.cnndm-src.json"
    qags_trg_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset500/prd.qst{n_qsts_per_doc}-ckptbest-gen.cnndm-gen.json"

elif exp_name == "subset1000":
    exp_d = subset1000_data
    n_qsts_per_doc = 5
    qags_src_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset1000/prd.qst{n_qsts_per_doc}-ckptbest-gen.cnndm-src.json"
    qags_trg_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset1000/prd.qst{n_qsts_per_doc}-ckptbest-gen.cnndm-gen.json"

elif exp_name == "xsum-subset1000":
    exp_d = xsum_subset1000_data
    n_qsts_per_doc = 6
    qags_src_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased/squad_v2_0/06-25-2019-v2_0/xsum-random1000/{mdl}/prd.qst{n_qsts_per_doc}-src.xsum-gen.json"
    qags_trg_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased/squad_v2_0/06-25-2019-v2_0/xsum-random1000/{mdl}/prd.qst{n_qsts_per_doc}-gen.xsum-gen.json"
else:
    exp_d = subset100_data
    n_qsts_per_doc = 10
    qags_src_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset/prd.qst{n_qsts_per_doc}-gen.cnndm-src.json"
    qags_trg_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset/prd.qst{n_qsts_per_doc}-gen.cnndm-gen.json"




#extract_src_trg_gen_from_fseq_log()
#extract_questions_and_write_jsonl()
#aggregate_questions()


#extract_subset()
#align_summaries()
#prepare_parlai_data()



compute_correlations_with_human(turk_files=exp_d[mdl],
                                ref_file=exp_d["ref"],
                                hyp_file=exp_d["hyp"][mdl],
                                mdl=mdl,
                                qags_src_file=qags_src_file,
                                qags_trg_file=qags_trg_file,
                                n_qsts_per_doc=n_qsts_per_doc
                                )


#mturk_posthoc()
