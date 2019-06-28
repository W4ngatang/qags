""" Evaluate answer spans """
import sys
import json
import random
import argparse

from sklearn.metrics import f1_score
import editdistance
def _em(x, y):
    """ exact match """
    return int(x == y)

def _f1(x, y):
    """ token level f1 """
    xs = x.split()
    ys = y.split()
    pcs = sum([1 for y in ys if y in xs]) / len(ys)
    rcl = sum([1 for x in xs if x in ys]) / len(xs)
    if pcs + rcl == 0:
        return 0
    else:
        return 2 * (pcs * rcl) / (pcs + rcl)

def _ed(x, y):
    """ edit distance """
    return editdistance.eval(x, y)

def evaluate(tgts, prds, metric_name="em"):
    """

    args:
        - tgt_anss: Target answers, usually predictions on full source, to be evaluated against
        - prd_anss: Answers to be evaluated

    returns:
        - average score
    """

    n_exs = len(tgts)
    total_score = 0.
    if metric_name == "em":
        metric = _em
    elif metric_name == "f1":
        metric = _f1
    elif metric_name == "ed":
        metric = _ed
    else:
        raise ValueError(f"Metric {metric_name} not found!")

    good_exs, bad_exs = [], []

    for ex_id, (tgt, prd) in enumerate(zip(tgts, prds)):
        score = metric(tgt, prd)
        total_score += score
        if metric_name == "em" and score == 1:
            good_exs.append(ex_id)
        elif metric_name == "em" and score == 0:
            bad_exs.append(ex_id)
        elif metric_name == "em":
            raise ValueError("Weird EM value found")

    return total_score / n_exs, good_exs, bad_exs

def load_data(data_file):
    """ """
    data = json.load(open(data_file, encoding="utf-8"))
    return data

def align_ans(srcs, trgs):
    """ """
    assert srcs.keys() == trgs.keys()
    src_ans = list(srcs.values())
    trg_ans = list(trgs.values())
    return src_ans, trg_ans

def main(arguments):
    parser = argparse.ArgumentParser(description='Evaluate answer outputs from pytorch_pretrained_bert models')
    parser.add_argument('--source-ans-file', type=str, default='/checkpoint/wangalexc/ppb/bert-base-uncased/squad_v2_0/06-25-2019-v2/predictions.cnndm-sources.source.json')
    parser.add_argument('--target-ans-file', type=str, default=None)
    parser.add_argument('--generation-ans-file', type=str, default=None)
    args = parser.parse_args()

    srcs = load_data(args.source_ans_file)
    if args.target_ans_file is not None:
        trgs = load_data(args.target_ans_file)
        src_ans, trg_ans = align_ans(srcs, trgs)
        for metric in ['em', 'f1', 'ed']:
            tgt_score, good_tgt, bad_tgt = evaluate(tgts=src_ans, prds=trg_ans, metric_name=metric)
            print(f"Tgt {metric}: {tgt_score}")

    if args.generation_ans_file is not None:
        gen_ans = load_data(args.generation_ans_file)
        src_ans, gen_ans = align_ans(srcs, gens)
        for metric in ['em', 'f1', 'ed']:
            gen_score, good_gen, bad_gen = evaluate(tgts=src_ans, prds=gen_ans, metric_name=metric)
            print(f"Gen {metric}: {gen_score}")

    src_ans = load_data(args.source_ans_file)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
