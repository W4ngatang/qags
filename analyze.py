""" Do various analysis things """
import os
import sys
import json
import random
import argparse

import numpy as np

from utils import load_txt

N_SAMPLE = 100

def sample_examples(verbose=True):
    qa_mdl = "bert-large-uncased-whole-word-masking"
    gen_mdl = "tfmr"
    qst_src = "gen"
    txt_ctx = "gen"
    src_qst_file = f"data/{gen_mdl}/qst-{qst_src}.cnndm-src.json"
    trg_qst_file = f"data/{gen_mdl}/qst-{qst_src}.cnndm-{txt_ctx}.json"
    src_ans_file = f"/checkpoint/wangalexc/ppb/{qa_mdl}/squad_v2_0/06-25-2019-v2_0/{gen_mdl}/prd.qst-{qst_src}.cnndm-src.json"
    trg_ans_file = f"/checkpoint/wangalexc/ppb/{qa_mdl}/squad_v2_0/06-25-2019-v2_0/{gen_mdl}/prd.qst-{qst_src}.cnndm-{txt_ctx}.json"

    out_dir = f"."
    match_file = os.path.join(out_dir, f"qst-{qst_src}.matched.txt")
    mismatch_file = os.path.join(out_dir, f"qst-{qst_src}.mismatched.txt")

    src_qsts = json.load(open(src_qst_file, encoding="utf-8"))['data']
    trg_qsts = json.load(open(trg_qst_file, encoding="utf-8"))['data']
    src_anss = json.load(open(src_ans_file, encoding="utf-8"))
    trg_anss = json.load(open(trg_ans_file, encoding="utf-8"))

    idxs = src_anss.keys()
    assert trg_anss.keys() == src_anss.keys()
    # find all matching and non matching answers
    match_idxs, nonmatch_idxs = [], []
    for idx in idxs:
        src_ans = src_anss[idx]
        trg_ans = trg_anss[idx]
        if src_ans == trg_ans:
            match_idxs.append(idx)
        else:
            nonmatch_idxs.append(idx)

    # sample some matching and non-matching answers
    sample_match = random.sample(match_idxs, N_SAMPLE)
    sample_nonmatch = random.sample(nonmatch_idxs, N_SAMPLE)

    if verbose:
        print("***** Matching answers *****")
    with open(match_file, "w", encoding="utf-8") as match_fh:
        for idx in sample_match:
            qst_idx = int(int(idx) / 5)  # 5 questions per passage
            qst = src_qsts[qst_idx]['paragraphs'][0]['qas'][0]['question']
            src = src_qsts[qst_idx]['paragraphs'][0]['context']
            trg = trg_qsts[qst_idx]['paragraphs'][0]['context']

            match_fh.write(f"Example {idx}\n")
            match_fh.write(f"Question: {qst}\n")
            match_fh.write(f"Answer: {src_anss[idx]}\n")
            match_fh.write(f"Source text: {src}\n")
            match_fh.write(f"Target text: {trg}\n")
            match_fh.write("\n")

            if verbose:
                print(f"Question: {qst}")
                print(f"Answer: {src_anss[idx]}")
                print(f"Source text: {src}")
                print(f"Target text: {trg}")
                print()

    if verbose:
        print("***** Non-matching answers *****")
    with open(mismatch_file, "w", encoding="utf-8") as mismatch_fh:
        for idx in sample_nonmatch:
            qst_idx = int(int(idx) / 5)  # 5 questions per passage
            qst = src_qsts[qst_idx]['paragraphs'][0]['qas'][0]['question']
            src = src_qsts[qst_idx]['paragraphs'][0]['context']
            trg = trg_qsts[qst_idx]['paragraphs'][0]['context']

            mismatch_fh.write(f"Example {idx}\n")
            mismatch_fh.write(f"Question: {qst}\n")
            mismatch_fh.write(f"Source answer: {src_anss[idx]}\n")
            mismatch_fh.write(f"Target answer: {trg_anss[idx]}\n")
            mismatch_fh.write(f"Source text: {src}\n")
            mismatch_fh.write(f"Target text: {trg}\n")
            mismatch_fh.write("\n")

            if verbose:
                print(f"Question: {qst}")
                print(f"Answer using source: {src_anss[idx]}")
                print(f"Answer using target: {trg_anss[idx]}")
                print()
                print(f"Source text: {src}\n")
                print(f"Target text: {trg}\n")

def _length(txt):
    """ """
    return len(txt)

def _get_ngrams(toks, ngram_size):
    """ """
    assert ngram_size > 0, f"Invalid ngram size: {ngram_size}"
    ngrams = zip(*[toks[i:] for i in range(ngram_size)])
    return ngrams


def _ngram_overlap(txt, ref, max_n=4):
    """ Compute n-gram overlap (precision and recall wrt ref) up to max_n

    args:
        - txt (List[str]): tokenized / split sentence
        - ref (List[str]): tokenized / split sentence
        - max_n (int): max ngram size

    returns:
        - ngrams (List[float]): list of ngram overlap ?
    """

    rcls, pcss = [], []
    for ngram_size in range(1, max_n + 1):
        txt_ngrams = list(_get_ngrams(txt, ngram_size))
        ref_ngrams = list(_get_ngrams(ref, ngram_size))
        rcl = sum([1 for ngram in ref_ngrams if ngram in txt_ngrams]) / len(ref_ngrams)
        pcs = sum([1 for ngram in txt_ngrams if ngram in ref_ngrams]) / len(txt_ngrams)
        rcls.append(rcl)
        pcss.append(pcs)
    return rcls, pcss

def compute_text_stats(stat_names, txts, refs, max_n=4):
    """ For all txts, compute stats, possibly using refs .

    args:

    returns:
        - stat2val (Dict[str: float]): dict mapping stat names to
            stat value averaged over all txts
    """

    if isinstance(txts[0], str):
        txts = [txt.split() for txt in txts]
        refs = [ref.split() for ref in refs]

    stat2val = {}
    for stat_name in stat_names:
        if stat_name == "length":
            stats = []
            for txt_idx, txt in enumerate(txts):
                stat = _length(txt)
                stats.append(stat)
            stats = np.array(stats)
            stat2val[f"{stat_name}-mean"] = stats.mean()
            stat2val[f"{stat_name}-std"] = stats.std()

        elif stat_name == "ngram-overlap":
            rcls, pcss = [], []
            for txt_idx, txt in enumerate(txts):
                rcl, pcs = _ngram_overlap(txt, refs[txt_idx], max_n)
                rcls.append(rcl)
                pcss.append(pcs)

            rcls = zip(*rcls)
            pcss = zip(*pcss)

            for i, (ngram_rcl, ngram_pcs) in enumerate(zip(rcls, pcss)):
                ngram_rcl = np.array(ngram_rcl)
                ngram_pcs = np.array(ngram_pcs)

                stat2val[f"{i+1}-gram_recall-mean"] = 100 * ngram_rcl.mean()
                stat2val[f"{i+1}-gram_precision-mean"] = 100 * ngram_pcs.mean()
                stat2val[f"{i+1}-gram_recall-std"] = 100 * ngram_rcl.std()
                stat2val[f"{i+1}-gram_precision-std"] = 100 * ngram_pcs.std()

    for stat_name, stat_val in stat2val.items():
        print(f"{stat_name}: {stat_val}")
    print(" & ".join([f"{v:.2f}" for v in stat2val.values()]))
    print()
    return stat2val

def main(arguments):

    #mdls = ["tfmr", "lstm", "lstmsmall", "lstmsmalltied"]
    txt_files = [f"/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/{mdl}-beam10.cnndm.test.txt" for mdl in mdls]
    ref_file = "/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/src.cnndm.test.txt"

    #txt_files = [f"/private/home/wangalexc/data/cnndailymail/fseq/gen.cnndm.test.txt"]
    #ref_file = f"/private/home/wangalexc/data/cnndailymail/fseq/src.cnndm.test.txt"

    refs = load_txt(ref_file)
    for txt_file in txt_files:
        txts = load_txt(txt_file)
        print(f"Computing stats for {txt_file}")
        stats = compute_text_stats(["length", "ngram-overlap"], txts, refs)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
