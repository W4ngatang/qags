""" Evaluate answer spans """
import re
import sys
import json
import string
import random
import argparse
import editdistance
from collections import Counter

import numpy as np

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


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def f1_score(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def edit_distance_score(prediction, ground_truth):
    """ edit distance """
    return editdistance.eval(normalize_answer(prediction), normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def aggregate_examples(scores, n_qst_per_doc=5):
    """Jank way to aggregate questions across examples.
    Right now (7/18/19), questions by examples are grouped together.
    """
    assert len(scores) % n_qst_per_doc == 0, "Number of questions invalid"
    n_doc = int(len(scores) / n_qst_per_doc)

    agg_scores = []
    for i in range(n_doc):
        agg_score = sum(scores[i * n_qst_per_doc : (i + 1) * n_qst_per_doc]) / n_qst_per_doc
        agg_scores.append(agg_score)

    return agg_scores


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

        # tracking goold + bad EM examples
        if metric_name == "em" and score == 1:
            good_exs.append(ex_id)
        elif metric_name == "em" and score == 0:
            bad_exs.append(ex_id)
        elif metric_name == "em":
            raise ValueError("Weird EM value found")

    scores = aggregate_examples(scores)

    scores = np.array(scores)
    mean = scores.mean()
    std = scores.std()
    if metric_name != "ed":
        mean *= 100.
        std *= 100.

    return mean, std, good_exs, bad_exs

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
    parser.add_argument('--source-ans-file', type=str, default=None)
    parser.add_argument('--target-ans-file', type=str, default=None)
    args = parser.parse_args()

    srcs = load_data(args.source_ans_file)
    if args.target_ans_file is not None:
        trgs = load_data(args.target_ans_file)
        src_ans, trg_ans = align_ans(srcs, trgs)
        for metric in ['em', 'f1', 'ed']:
            tgt_mean, tgt_std, good_tgt, bad_tgt = evaluate(tgts=src_ans, prds=trg_ans, metric_name=metric)
            print(f"Tgt {metric}: mean {tgt_mean}, std {tgt_std}")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
