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

    data = [json.loads(d) for d in open(data_file, encoding="utf-8")]
    qsts = []
    srcs, tgts, gens = [], [], []
    s_ans, t_ans, g_ans = [], [], []
    for datum in data:
        srcs.append(datum["source"])
        tgts.append(datum["target"])
        gens.append(datum["hypotheses"][0])

        s_ans.append(datum["src_ans"])
        t_ans.append(datum["trg_ans"])
        g_ans.append(datum["gen_ans"])

        qsts.append(datum["questions"][0])
    assert len(srcs) == len(tgts) == len(gens)
    return qsts, s_ans, t_ans, g_ans, srcs, tgts, gens

def print_samples(ex_idxs, par, ans, qst, tgt, n_samples=5):
    n_samples = min(len(ex_idxs), n_samples)
    sample_idxs = random.sample(ex_idxs, n_samples)
    for idx in sample_idxs:
        print(f"Ex {idx}: {par[idx]}")
        print(f"Qst: {qst[idx]}")
        print(f"\"Tgt\": {tgt[idx]}")
        print(f"Ans: {ans[idx]}\n")

def main(arguments):
    parser = argparse.ArgumentParser(description='Evaluate answers')
    parser.add_argument('--metric', type=str, default='em',
                        choices=['em', 'f1', 'ed'])
    parser.add_argument('--data-file', type=str, default='data/bidaf_outs.jsonl')
    args = parser.parse_args()

    qsts, s_ans, t_ans, g_ans, srcs, tgts, gens = load_data(args.data_file)
    tgt_score, good_tgt, bad_tgt = evaluate(tgts=s_ans, prds=t_ans, metric_name=args.metric)
    gen_score, good_gen, bad_gen = evaluate(tgts=s_ans, prds=g_ans, metric_name=args.metric)
    print(f"Tgt score: {tgt_score}")
    print(f"Gen score: {gen_score}")
    print()

    print("Good target answers")
    print_samples(good_tgt, par=tgts, ans=t_ans, qst=qsts, tgt=s_ans)
    print("Bad target answers")
    print_samples(bad_tgt, par=tgts, ans=t_ans, qst=qsts, tgt=s_ans)

    print("Good generation answers")
    print_samples(good_gen, par=gens, ans=g_ans, qst=qsts, tgt=s_ans)
    print("Bad generation answers")
    print_samples(bad_gen, par=gens, ans=g_ans, qst=qsts, tgt=s_ans)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
