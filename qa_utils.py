""" Evaluate answer spans """
import os
import re
import sys
import json
import string
import random
import argparse
import editdistance
from collections import Counter
import ipdb

import numpy as np
from scipy.stats import pearsonr, spearmanr


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

def load_correctness(data_file):
    """ Load file with correctness labels per summary
    Currently very ad hoc """

    return list(map(lambda x: float(x.strip()), open(data_file).readlines()))




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


def aggregate_questions_from_txt():
    """ Extract questions generated from src, trg, and gen
    with the corresponding field from fseq logs (one log/txt) and write to jsonl.
    Each fseq log should have the txt field as 'source' (S)
    and the questions as generated 'hypotheses' (H) """

    # Parameters
    data = 'wikinews'
    gen_mdl = 'bart'
    subset = '120519' # NOTE(Alex): IF IT'S 250, IT SHOULD BE 6250!
    n_exs = 100
    if data == "cnndm":
        data_dir = f"{DATA_DIR}/cnndailymail/fseq"
    elif data == "xsum":
        data_dir = f"{DATA_DIR}/xsum"
    elif data == "falke-sent-rerank":
        data_dir = f"{DATA_DIR}/falke-correctness/sent-rerank"
    elif data == "wikinews":
        data_dir = f"{DATA_DIR}/wikinews"

    dataset = f'{data}-{subset}'
    qg_model = 'qg-newsqa-ans'
    bert_version = 'bert-large-uncased'
    n_qsts = 20 # n questions we actually want to use
    n_gen_qsts = 10 # n questions generated per doc
    n_ans = 10 # n answer candidates
    use_all_qsts = False # use all qsts, mostly if we want answers to our questions
    use_act_anss = True # use actual answer (filter if actual answer is empty)
    use_exp_anss = False # use expected answer (filter if actual answer doesn't match)
    beam = 10
    topk = 0
    topp = 0
    diverse = 0
    reverse_prob = False
    #dec_method = 'nhyps25.beam25.diverse25'
    dec_method = 'nhyps10.beam10.diverse10'
    #dec_method = ''

    # Some sanity checks
    if use_all_qsts:
        assert n_qsts == n_gen_qsts, f"Only using {n_qsts} of {n_gen_qsts} questions!"

    # Original texts
    if n_ans > 0:
        dataset = f'{dataset}-{n_ans}ans'
        data_subdir = f'{subset}-{n_ans}ans-{dec_method}' if dec_method else f'{subset}-{n_ans}ans'
        src_txt_file = f"{data_dir}/{data_subdir}/test.src.txt"
        src_w_trg_txt_file = f"{data_dir}/{data_subdir}/test.src_w_trg.txt" if data in ["xsum", "wikinews"] else None
        gen_txt_file = f"{data_dir}/{data_subdir}/test.{gen_mdl}.txt"
        src_ans_file = f"{data_dir}/{data_subdir}/test.src_ans.txt"
        gen_ans_file = f"{data_dir}/{data_subdir}/test.{gen_mdl}_w_ans.txt"
    else:
        # NOTE(Alex): these aren't abstracted / generalized
        src_txt_file = f"{data_dir}/{subset}/src2bart/raw/test.src"
        gen_txt_file = f"{data_dir}/{subset}/bart2src/raw/test.src"

    dataset = f'{dataset}-{dec_method}' if dec_method else dataset

    # Files containing all generated questions
    if use_all_qsts:
        qst_prefix = "qstall"
    elif use_exp_anss:
        qst_prefix = f"qst_w_match{n_qsts}{bert_version}"
    elif use_act_anss:
        qst_prefix = f"qst_w_ans{n_qsts}{bert_version}"
    else:
        qst_prefix = f"qst{n_qsts}"

    if topk > 0:
        dec_opt = f'topk{topk}'
    elif topp > 0:
        dec_opt = f'topp{topp}'
    elif diverse:
        dec_opt = f'beam{beam}.diverse{diverse}'
    else:
        dec_opt = f'beam{beam}'
    src_qst_file = f"{CKPT_DIR}/bart/{dataset}/src2{gen_mdl}/{qg_model}/gens.nhyps{n_gen_qsts}.lenpen1.0.{dec_opt}.txt"
    gen_qst_file = f"{CKPT_DIR}/bart/{dataset}/{gen_mdl}2src/{qg_model}/gens.nhyps{n_gen_qsts}.lenpen1.0.{dec_opt}.txt"
    src_prob_file = f"{CKPT_DIR}/bart/{dataset}/src2{gen_mdl}/{qg_model}/gens.nhyps{n_gen_qsts}.lenpen1.0.{dec_opt}.prob"
    gen_prob_file = f"{CKPT_DIR}/bart/{dataset}/{gen_mdl}2src/{qg_model}/gens.nhyps{n_gen_qsts}.lenpen1.0.{dec_opt}.prob"
    dec_opt = f'{dec_opt}'
    src_prd_file = f""
    gen_prd_file = f"{CKPT_DIR}/ppb/{bert_version}/squad_v2_0/06-25-2019-v2_0/{dataset}/bart/prd.qstall-gen-{qg_model}-{dec_opt}.{dataset}-gen.json"

    files = {
             "src": {"txt": src_txt_file, "qst": src_qst_file, "prb": src_prob_file, "prd": src_prd_file},
             "gen": {"txt": gen_txt_file, "qst": gen_qst_file, "prb": gen_prob_file, "prd": gen_prd_file},
            }

    out_dir = f"{PROJ_DIR}/data/{data}/{subset}"
    if n_ans > 0:
        out_dir = f"{out_dir}-{n_ans}ans"
        n_gen_qsts *= n_ans
        files["src"]["ans"] = src_ans_file
        files["gen"]["ans"] = gen_ans_file
    out_dir = f"{out_dir}-{dec_method}" if dec_method else out_dir
    out_dir = f"{out_dir}-reverse" if reverse_prob else out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"Reading data from {src_qst_file} and {gen_qst_file}, saving to {out_dir}")

    all_txts, all_qsts = {}, {}
    for txt_fld, field_files in files.items():
        txts = load_txt(field_files["txt"])
        all_txts[txt_fld] = txts

        if txt_fld == "src" and src_w_trg_txt_file is not None:
            txts = load_txt(src_w_trg_txt_file)
            all_txts["src_w_trg"] = txts

        if txt_fld != "gen":
            continue
        qsts = load_txt(field_files["qst"])
        prbs = [float(f) for f in load_txt(field_files["prb"])]
        anss = load_txt(field_files["ans"]) if ("ans" in field_files and use_exp_anss) else list()
        if "prd" in field_files and use_act_anss:
            raw_prds = json.load(open(field_files["prd"]))
            prds = [raw_prds[str(i)] for i in range(len(raw_prds))]
        else:
            prds = list()
        all_qsts[txt_fld] = (qsts, prbs, anss, prds)

    # for each (question from a source x txt source) pair,
    # build the data then write out a SQuAD format file
    bookkeep = {k: {'idxs': list(), 'min': n_gen_qsts, 'n_below': 0, 'counts': list()} for k in \
                 ['n_clean_qsts', 'n_qsts_w_ans', 'n_qsts_w_match_ans']}
    for qst_src in all_qsts:
        if qst_src != "gen":
            continue

        qsts, prbs, anss, prds = all_qsts[qst_src]
        all_clean_qsts = list()

        # Filter questions
        for i in tqdm(range(n_exs), desc="Filtering questions"):
            cand_qsts = qsts[(i * n_gen_qsts): ((i + 1) * n_gen_qsts)]
            cand_prbs = prbs[(i * n_gen_qsts): ((i + 1) * n_gen_qsts)]
            cand_anss = anss[(i * n_ans): ((i + 1) * n_ans)] if anss else None
            cand_prds = prds[(i *n_gen_qsts): ((i + 1) * n_gen_qsts)] if prds else None
            if not use_all_qsts:
                ret = filter_qsts(cand_qsts, n_qsts,
                                  prbs=cand_prbs, reverse_prob=reverse_prob,
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

            # Bookkeeping for questions
            for k, v in ret.items():
                if not isinstance(v, int):
                    continue
                if v < bookkeep[k]['min']:
                    bookkeep[k]['min'] = v
                    bookkeep[k]['idxs'] = [i]
                elif v == bookkeep[k]['min']:
                    bookkeep[k]['idxs'].append(i)
                if v < n_qsts:
                    bookkeep[k]['n_below'] += 1
                bookkeep[k]['counts'].append(v)

        # Construct data in SQuAD-like format
        for txt_fld in all_txts:
            if use_all_qsts and txt_fld != "gen":
                # case where we want to get answers for all our questions
                # and we want to just use the generations to do that,
                # assuming we generated from generations
                continue

            txts = all_txts[txt_fld]

            raw_data = {}
            for i in tqdm(range(n_exs), desc="Formatting data"):
                txt = txts[i * n_ans].split()
                clean_qsts = all_clean_qsts[i]
                raw_data[i] = {txt_fld: txt, "hypotheses": clean_qsts}

            data = format_squad(raw_data, context=txt_fld, ctx_split=True)

            out_file = f"{out_dir}/{qst_prefix}-{qst_src}-{qg_model}-{dec_opt}.{dataset}-{txt_fld}.json"
            print(f"Writing to {out_file}")
            json.dump(data, open(out_file, "w", encoding="utf-8"))

    for k, v in bookkeep.items():
        if not v['counts']:
            continue
        counts = np.array(v['counts'])
        print(f"{k}: ")
        print(f"\t{len(v['idxs'])} exs with min {v['min']} (idxs: {list(set(v['idxs']))})")
        print(f"\t{v['n_below']} exs w/ fewer than {n_qsts} clean questions!")
        print(f"\tmean: {np.mean(counts)}")
        print(f"\tmedian: {np.median(counts)}")
        print(f"\tmax: {np.max(counts)}")
        print(f"\tmin: {np.min(counts)}")


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
    parser.add_argument("--command", choices=["compute-qags"], description="Function to perform")
    parser.add_argument('--source-ans-file', type=str)
    parser.add_argument('--target-ans-file', type=str)
    parser.add_argument('--n-qsts-per-doc', type=int, default=5)
    parser.add_argument('--ans-similarity-fn', choices=["em", "f1"], default="f1")
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--correctness-file', type=str, default=None)
    args = parser.parse_args()

    if args.command == "format-data":
        aggregate_questions_from_txt()
    elif args.command == "compute-qags":
        qags_scores = get_qags_scores(args.src_ans_file, args.trg_ans_file, args.ans_similarity_fn)
        with open(os.path.join(args.out_dir, "qags_scores.txt", "w") as out_fh:
            for score in qags_scores:
                out_fh.write(f"{score}\n")


    #srcs = load_data(args.source_ans_file)
    #trgs = load_data(args.target_ans_file)
    #src_ans, trg_ans = align_ans(srcs, trgs)
    #count_noans(src_ans, trg_ans)
    #for metric in ['em', 'f1', 'ed']:
    #    scores, good_tgt, bad_tgt = evaluate(tgts=src_ans, prds=trg_ans,
    #                                         metric_name=metric,
    #                                         n_qsts_per_doc=args.n_qsts_per_doc)
    #    print(f"Tgt {metric}: mean {np.mean(scores)}, std {np.std(scores)}")
    #    if args.outdir is not None:
    #        json.dump(scores, open(os.path.join(args.outdir, f"{metric}_scores.json"), "w"))

    #if args.correctness_file is not None:
    #    correctness = load_correctness(args.correctness_file)
    #    corr, pval = pearsonr(scores[:50], correctness)
    #    print(f"Pearson corr wrt {args.correctness_file}: {corr} (p-val {pval})")
    #    #print(f"# incorrect: {sum(1 - c for c in correctness) / len(correctness)}")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
