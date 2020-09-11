""" Evaluate answer spans """
import os
import re
import sys
import json
import string
import random
import argparse
import editdistance
from tqdm import tqdm
from collections import Counter

import numpy as np
from scipy.stats import pearsonr, spearmanr
from utils import write_data, write_jsonl, write_txt, \
                  process, print_samples, format_squad, \
                  filter_line_fseq, parse_generation, \
                  load_txt, load_json

import ipdb


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


def count_noans(src_anss, trg_anss):
    """ """
    n_src, n_trg = len(src_anss), len(trg_anss)
    src_unans, trg_unans, both_unans = 0, 0, 0
    for src_ans, trg_ans in zip(src_anss, trg_anss):
        if src_ans == "" and trg_ans == "":
            both_unans += 1
        if src_ans == "":
            src_unans += 1
        if trg_ans == "":
            trg_unans += 1
    percent_src_noans = src_unans / n_src
    percent_trg_noans = trg_unans / n_trg
    print(f"% source no answer: {percent_src_noans} ({src_unans} / {n_src}) ")
    print(f"% target no answer: {percent_trg_noans} ({trg_unans} / {n_trg}) ")
    print(f"# both no answer: {both_unans}")
    return percent_src_noans, percent_trg_noans, both_unans


def aggregate_examples(scores, n_qsts_per_doc=5):
    """Jank way to aggregate questions across examples.
    Right now (7/18/19), questions by examples are grouped together.
    """
    assert len(scores) % n_qsts_per_doc == 0, "Number of questions invalid"
    n_doc = int(len(scores) / n_qsts_per_doc)

    agg_scores = []
    for i in range(n_doc):
        agg_score = sum(scores[i * n_qsts_per_doc : (i + 1) * n_qsts_per_doc]) / n_qsts_per_doc
        agg_scores.append(agg_score)

    return agg_scores


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
        - act_anss: actual answers, e.g. from a QA model (optional)
    """

    qsts_and_prbs = zip(qsts, prbs)
    if act_anss is not None:
        qsts_and_prbs = [(q, p) for q, p, a in zip(qsts, prbs, act_anss) if a]
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


def aggregate_questions_from_txt(out_dir,
                                 src_txt_file,
                                 gen_txt_file,
                                 gen_qst_file,
                                 gen_prob_file=None,
                                 gen_ans_file=None,
                                 gen_prd_file=None,
                                 src_w_trg_txt_file=None,
                                 use_all_qsts=False, use_act_anss=False, use_exp_anss=False,
                                 n_gen_qsts=10, n_ans=10, n_qsts=20):
    """ Extract questions generated from src, trg, and gen
    with the corresponding field from fseq logs (one log/txt) and write to jsonl.
    Each fseq log should have the txt field as 'source' (S)
    and the questions as generated 'hypotheses' (H)

    args:
        - src_txt_file: txt file or source inputs (e.g. articles for summarization)
            - src_w_trg_txt_file (optional): special src inputs with the trg re-appended for XSUM
        - gen_txt_file: txt file of model-generated targets (e.g. summaries for summarization)
        - gen_qst_file: txt file of questions generated conditioned on src/gen
        - gen_prob_file: txt file of {src/gen} question probabilities according to QG model
        - gen_prd_file (optional): txt file of answers predicted by QA model on src/gen_qst_file

        n_ans: the number of answer candidates per text
        n_gen_qsts: the number of questions generated per (text, answer) pair
        n_qsts: the number of questions to use for each example
        use_all_qsts: use all questions
        use_act_anss: filter out [NO_ANS] questions
        use_exp_anss: filter out questions where prediction doesn't match expected answer
    """

    assert not (use_exp_anss and (gen_ans_file is None)), "Trying to use expected answers, but not provided any!"
    assert not (use_act_anss and (gen_ans_file is None)), "Trying to use predicted answers, but not provided expected answers!"
    assert not (use_act_anss and (gen_prd_file is None)), "Trying to use predicted answers, but not provided any!"

    files = {
             "src": {"txt": src_txt_file},
             "gen": {"txt": gen_txt_file, "qst": gen_qst_file, "prb": gen_prob_file, "ans": gen_ans_file, "prd": gen_prd_file},
            }

    # the number of original examples (not counting answer candidates)
    n_exs = None
    # number of total generated questions per example (across answer candidates and generated questions)
    n_qsts_per_ex = n_ans * n_gen_qsts

    # load all data
    all_txts, all_qsts = {}, {}
    for txt_fld, field_files in files.items():
        txts = load_txt(field_files["txt"])
        all_txts[txt_fld] = txts
        if txt_fld == "src" and src_w_trg_txt_file is not None:
            txts = load_txt(src_w_trg_txt_file)
            all_txts["src_w_trg"] = txts

        if n_exs is None: # infer number of examples
            n_exs = len(txts)
        else:
            assert len(txts) == n_exs, "Different numbers of txts detected! Expected {n_exs} but found {len(txts)} for {txt_fld}."

        # load questions, probabilities, (expected) answers only based on generation
        if txt_fld != "gen":
            continue
        qsts = load_txt(field_files["qst"])
        prbs = [float(f) for f in load_txt(field_files["prb"])] #if field_files["prb"] is not None else list()
        anss = load_txt(field_files["ans"]) if use_exp_anss else []
        # optionally load QA model predictions
        if use_act_anss:
            raw_prds = json.load(open(field_files["prd"]))
            prds = [raw_prds[str(i)] for i in range(len(raw_prds))]
        else:
            prds = list()
        all_qsts[txt_fld] = (qsts, prbs, anss, prds)
    print(f"Formatting QA data for {n_exs} examples, filtering {n_qsts_per_ex} questions per example to {n_qsts}")

    # build the data then write out a SQuAD format file
    # dummy iterator in case we want to condition questions on something else outside of gen
    for qst_src in all_qsts:
        qsts, prbs, anss, prds = all_qsts[qst_src]
        all_clean_qsts = []

        # Filter questions
        # Extract questions assuming there's a constant number per example and in order
        for i in tqdm(range(n_exs), desc="Filtering questions"):
            cand_qsts = qsts[(i * n_qsts_per_ex): ((i + 1) * n_qsts_per_ex)]
            cand_prbs = prbs[(i * n_qsts_per_ex): ((i + 1) * n_qsts_per_ex)]
            cand_anss = anss[(i * n_ans): ((i + 1) * n_ans)] if anss else None
            cand_prds = prds[(i * n_qsts_per_ex): ((i + 1) * n_qsts_per_ex)] if prds else None
            if not use_all_qsts:
                ret = filter_qsts(cand_qsts, n_qsts,
                                  prbs=cand_prbs, reverse_prob=False,
                                  exp_anss=cand_anss, act_anss=cand_prds)
            else:
                ret = {
                       'qsts': cand_qsts,
                       'n_clean_qsts': len(cand_qsts),
                       'n_qsts_w_ans': None,
                       'n_qsts_w_match_ans': None,
                      }
            clean_qsts = ret['qsts']
            for qst in clean_qsts:
                assert not isinstance(qst, list), "List instead of string detected!"
            all_clean_qsts.append(clean_qsts)

        # Construct data in SQuAD-like format, using both src (article) and gen (model generation) as context
        for txt_fld in all_txts:
            if use_all_qsts and txt_fld != "gen":
                # case where we want to get answers for all our questions
                # and we want to just use the generations to do that, assuming we generated from generations
                continue
            txts = all_txts[txt_fld]
            raw_data = {}

            for i in tqdm(range(n_exs), desc="Formatting data"):
                if len(txts) == len(qsts):
                    txt = txts[i * n_ans].split()
                elif len(txts) < len(qsts):
                    assert len(qsts) / len(txts) == n_qsts_per_ex, \
                            f"Expected constant number of questions ({n_qsts_per_ex}) per example! Found {len(qsts)} total questions for {len(txts)} examples"
                    txt = txts[i].split()
                else:
                    raise IndexError("Number of questions should be weakly greater than number of examples!")
                clean_qsts = all_clean_qsts[i]
                raw_data[i] = {txt_fld: txt, "hypotheses": clean_qsts}

            data = format_squad(raw_data, context=txt_fld, ctx_split=True)
            out_file = f"{out_dir}/{txt_fld}.json"
            print(f"Writing to {out_file}")
            json.dump(data, open(out_file, "w", encoding="utf-8"))


def evaluate(tgts, prds, n_qsts_per_doc, metric_name="em"):
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

    scores = aggregate_examples(scores, n_qsts_per_doc)

    scores = np.array(scores)
    mean = scores.mean()
    std = scores.std()
    if metric_name != "ed":
        mean *= 100.
        std *= 100.

    return scores.tolist(), good_exs, bad_exs


def get_qags_scores(src_ans_file, trg_ans_file,
                    metric_name="em", n_qsts_per_doc=10):
    """Load answer files and compute similarity scores
    """
    srcs = load_data(src_ans_file)
    trgs = load_data(trg_ans_file)
    src_ans, trg_ans = align_ans(srcs, trgs)
    qags_scores, _,  _ = evaluate(tgts=src_ans, prds=trg_ans,
                                  n_qsts_per_doc=n_qsts_per_doc,
                                  metric_name=metric_name)
    return qags_scores


def main(arguments):
    parser = argparse.ArgumentParser(description='Evaluate answer outputs from pytorch_pretrained_bert models')
    parser.add_argument("-c", "--command", choices=["compute-qags", "format-qa-data"])
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)

    parser.add_argument('--src-txt-file', type=str,
                        help="Txt file containing a src example per line, corresponding with gen_txt_file")
    parser.add_argument('--src-w-trg-txt-file', type=str, default=None, help="special input for XSUM")
    parser.add_argument('--gen-txt-file', type=str,
                        help="Txt file containing a model-generated example per line, corresponding with src_txt_file")
    parser.add_argument('--gen-qst-file', type=str,
                        help="Txt file containing a gen-conditioned question per line, in the same order as {src/gen}_txt_file")
    parser.add_argument('--gen-prob-file', type=str, default=None,
                        help="Txt file containing probabilities of each question in gen_qst_file according to the QG model")
    parser.add_argument('--gen-ans-file', type=str, default=None,
                        help="Txt file containing expected answers of each question in gen_qst_file")
    parser.add_argument('--gen-prd-file', type=str, default=None,
                        help="Txt file containing predictions of QA model on questions in gen_qst_file")
    parser.add_argument('--n-ans-per-doc', type=int, default=10, help="Number of answer candidates per example")
    parser.add_argument('--n-gen-qsts', type=int, default=10, \
            help="Number of generated questions per (example, answer candidate) pair")
    parser.add_argument('--n-qsts-per-doc', type=int, default=5, \
            help="Number of questions to use per example, filtered down from n_ans_per_doc * n_gen_qsts")
    parser.add_argument('--use-all-qsts', action='store_true')
    parser.add_argument('--use-act-anss', action='store_true')
    parser.add_argument('--use-exp-anss', action='store_true')

    parser.add_argument('--source-ans-file', type=str)
    parser.add_argument('--target-ans-file', type=str)
    parser.add_argument('--ans-similarity-fn', choices=["em", "f1"], default="f1")
    args = parser.parse_args()


    if args.command == "format-qa-data":
        aggregate_questions_from_txt(args.out_dir,
                                     args.src_txt_file,
                                     args.gen_txt_file,
                                     args.gen_qst_file,
                                     args.src_w_trg_txt_file,
                                     args.gen_prob_file,
                                     args.gen_ans_file,
                                     args.gen_prd_file,
                                     n_ans=args.n_ans_per_doc, n_gen_qsts=args.n_gen_qsts, n_qsts=args.n_qsts_per_doc,
                                     use_all_qsts=args.use_all_qsts, use_act_anss=args.use_act_anss, use_exp_anss=args.use_exp_anss,
                                    )

    elif args.command == "compute-qags":
        qags_scores = get_qags_scores(args.src_ans_file, args.trg_ans_file, args.ans_similarity_fn)
        with open(os.path.join(args.out_dir, "qags_scores.txt", "w")) as out_fh:
            for score in qags_scores:
                out_fh.write(f"{score}\n")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
