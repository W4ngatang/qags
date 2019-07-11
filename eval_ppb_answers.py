""" Evaluate answer spans """
import re
import sys
import json
import string
import random
import argparse
import editdistance
from collections import Counter

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


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def edit_distance_score(prediction, ground_truth):
    """ edit distance """
    return editdistance.eval(normalize_answer(prediction), normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(tgts, prds, metric_name="em"):
    """

    args:
        - tgt_anss: Target answers, usually predictions on full source, to be evaluated against
        - prd_anss: Answers to be evaluated

    returns:
        - average score
    """

    n_exs = len(tgts)
    scores = []
    if metric_name == "em":
        metric = exact_match_score
    elif metric_name == "f1":
        metric = f1_score
    elif metric_name == "ed":
        metric = edit_distance_score
    else:
        raise ValueError(f"Metric {metric_name} not found!")

    good_exs, bad_exs = [], []

    for ex_id, (tgt, prd) in enumerate(zip(tgts, prds)):
        score = metric_max_over_ground_truths(metric, prd, [tgt])

        scores.append(score)
        if metric_name == "em" and score == 1:
            good_exs.append(ex_id)
        elif metric_name == "em" and score == 0:
            bad_exs.append(ex_id)
        elif metric_name == "em":
            raise ValueError("Weird EM value found")

    return 100. * sum(scores) / n_exs, good_exs, bad_exs

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
