""" Evaluate answer spans """
import sys
import json
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

    for tgt, prd in zip(tgts, prds):
        total_score += metric(tgt, prd)

    return total_score / n_exs

def load_data(data_file):
    """ """

    data = [json.loads(d) for d in open(data_file, encoding="utf-8")]
    srcs, tgts, gens = [], [], []
    for datum in data:
        srcs.append(datum["src_ans"])
        tgts.append(datum["trg_ans"])
        gens.append(datum["gen_ans"])
    assert len(srcs) == len(tgts) == len(gens)
    return srcs, tgts, gens

def main(arguments):
    parser = argparse.ArgumentParser(description='Evaluate answers')
    parser.add_argument('--metric', type=str, default='em',
                        choices=['em', 'f1', 'ed'])
    parser.add_argument('--data-file', type=str, default='data/bidaf_outs.jsonl')
    args = parser.parse_args()

    srcs, tgts, gens = load_data(args.data_file)
    tgt_score = evaluate(tgts=srcs, prds=tgts, metric_name=args.metric)
    gen_score = evaluate(tgts=srcs, prds=gens, metric_name=args.metric)
    print(f"Tgt score: {tgt_score}")
    print(f"Gen score: {gen_score}")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
