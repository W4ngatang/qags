""" Extract generations from fairseq outputs """

import os
import re
import ast
import json
import copy
import random
import itertools
from collections import defaultdict, Counter

import ipdb
import rouge
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from nltk.tokenize import sent_tokenize
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler

from utils import write_data, write_jsonl, write_txt, \
                  process, print_samples, format_squad, \
                  filter_line_fseq, parse_generation, \
                  load_txt, load_json
from eval_ppb_answers import evaluate, load_data, align_ans

N_SAMPLES = 5
ATTN_IDX = -2
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


# Extract questions
def extract_questions_and_write_jsonl():
    #data_file = "/checkpoint/wangalexc/fairseq/06-18-2019/questions-cnndm.sampling.out"
    #out_file = "/private/home/wangalexc/projects/qags/data/questions.sampling.jsonl"

    #data_file = "/checkpoint/wangalexc/fairseq/06-27-2019/questions.cnndm-target.out"
    #out_file = "/private/home/wangalexc/projects/qags/data/questions.cnndm-target.jsonl"
    #aux_file = "/private/home/wangalexc/projects/qags/data/questions.cnndm-summaries.jsonl"
    #swap_d = {"source": "target", "target": "source" } # will error as is

    data_file = "/checkpoint/wangalexc/fairseq/07-09-2019/questions-cnndm-summaries-full.out"
    out_file = "/private/home/wangalexc/projects/qags/data/questions.cnndm-summaries.jsonl"
    #aux_file = "/private/home/wangalexc/projects/qags/data/questions.cnndm-targets.jsonl"
    swap_d = {"source": "generation", "target": "source" }

    data = parse_generation(data_file)
    data = swap_fields(data, swap_d)
    write_jsonl(data, out_file, swap_fields)
    print_samples(data, n_samples=N_SAMPLES)
    print(f"Extracted questions from {data_file} and wrote to {out_file}")


def format_abstractive_qa():
    """ Format an extractive QA task as a freeform QA task """

    def _combine(x, y):
        """ Combine two texts, x and y """
        return f"{x} {y}"

    def _format_qa_split(data):
        """ """
        srcs, trgs = [], []

        for doc in data:
            for psg_d in doc["paragraphs"]:
                psg = psg_d["context"]
                for qst_d in psg_d["qas"]:
                    qst = qst_d["question"]
                    src = _combine(qst, psg)
                    srcs.append({"input": src})

                    anss = [a["text"] for a in qst_d["answers"]]
                    if len(anss) == 0: # question has no answer
                        ans = "<na>"
                    else:
                        ans = anss[0]
                    trgs.append({"target": ans})

        return srcs, trgs

    data_dir = "/private/home/wangalexc/data/squad/v2.0/original"
    out_dir = "/private/home/wangalexc/data/squad/v2.0/abstractive"
    split2file = {"train": "train.json",
                  "dev": "dev.json"
                  #"test": "test.json"
                 }

    for split_name, split_file in split2file.items():
        data = load_json(os.path.join(data_dir, split_file))["data"]
        srcs, trgs = _format_qa_split(data)
        write_data(srcs=srcs, trgs=trgs, out_dir=out_dir, out_prefix=split_name, out_format="jsonl")


def process_human_subset():

    label_map = {'Incorrect': 0, 'Unclear': 0, 'Correct': 1}
    txt_flds = ['src', 'trg', 'pgc', 'fan', 'bus']
    data_file = "data/aligned-subsets.json"
    data = json.load(open(data_file))

    for txt_fld in txt_flds:
        vals = list(data.values())
        txts = [v[txt_fld] for v in vals]
        if txt_fld in ['src', 'trg']:
            txts = [" ".join(ex) for ex in txts]
        else:
            txts = [" ".join([s['text'] for s in ex['sents'].values()]) for ex in txts]
            binary_scores = [label_map[v[txt_fld]['label']] for v in vals]
            count_scores = [sum([label_map[s['label']] for s in ex[txt_fld]['sents'].values()]) / len(ex[txt_fld]['sents']) for ex in vals]
        with open(f"data/labeled-subset/{txt_fld}-subset.txt", "w") as out_fh:
            for txt in txts:
                out_fh.write(f'{txt}\n')
        if txt_fld not in ['src', 'trg']:
            with open(f"data/labeled-subset/{txt_fld}-subset.scores.txt", "w") as out_fh:
                out_fh.write("\n".join(map(str, count_scores)))
            with open(f"data/labeled-subset/{txt_fld}-subset.binary-scores.txt", "w") as out_fh:
                out_fh.write("\n".join(map(str, binary_scores)))


JOIN_PUNCT = {'- lrb -': '-lrb-', '- rrb -': '-rrb-',
              'n \'t': 'n\'t', '\' s': '\'s'
             }
REPLACE_PUNCT = {'-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']', '#': '$'}
NO_LEADING_SPACE_PUNCT = ['.', ',', '\'s', '\'m', '\'ve', '\'d', '?', '!', 'n\'t', '\'re']
NO_TRAILING_SPACE_PUNCT = [' `', ' \'']


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

    def get_aligned_shortest(txts1, txts2, n_exs_to_search=1000):
        """Get alignment in txt2 of the shortest n_exs summaries in txts1
        """
        sorted_txts2 = sorted(enumerate(txts2), key=lambda x: len(x[1].split()))
        cnts1 = [Counter(t.split()) for t in txts1]
        cnts2 = [(i, Counter(t.split())) for i, t in sorted_txts2[:n_exs_to_search]]

        diffs = [[(i2, sum((c1 & c2).values())) for i2, c2 in cnts2] for c1 in cnts1]
        idxs2 = list()
        for diff in diffs:
            idx2, _ = max(diff, key=lambda x: x[-1])
            idxs2.append(idx2)

        def print_algn(idx):
            print(f"{idx}: {txts1[idx]}\n")
            print(f"{alignment[idx]}: {txts2[alignment[idx]]}\n")

        return idxs2

    n_exs = 500
    n_exs_to_search = 750

    ref_src_file = "/misc/vlgscratch4/BowmanGroup/awang/processed_data/cnndailymail/fseq/src.cnndm.test.txt"
    ref_trg_file = "/misc/vlgscratch4/BowmanGroup/awang/processed_data/cnndailymail/fseq/trg.cnndm.test.txt"
    # Load the sources that we'll use
    ref_srcs = load_txt(ref_src_file)
    ref_trgs = load_txt(ref_trg_file)
    sorted_ref_srcs_and_idxs = sorted(enumerate(ref_srcs), key=lambda x: len(x[1].split()))
    sorted_ref_idxs, sorted_ref_srcs = zip(*sorted_ref_srcs_and_idxs[:n_exs])
    sorted_ref_trgs = [ref_trgs[i] for i in sorted_ref_idxs]
    write_txt(sorted_ref_srcs, f"data/subset{n_exs}.src.txt")
    write_txt(sorted_ref_trgs, f"data/subset{n_exs}.trg.txt")

    mdl_files = {
                 "bus": ("data/all_bus/all_bus.trg.txt", "data/all_bus/all_bus.src.400words.txt"),
                 "fas": ("data/all_fas/all_fas_rerank.trg.txt", "/misc/vlgscratch4/BowmanGroup/awang/raw_data/cnndm_harvard/test.txt.src"),
                 "pgc": ("data/all_pgc/all_pgc.trg.txt", "data/all_pgc/all_pgc.src.txt"),
                }

    for mdl_name, (mdl_gen_file, mdl_ref_file) in mdl_files.items():
        print(f"Processing data for {mdl_name}")
        mdl_gens = load_txt(mdl_gen_file)
        mdl_refs = load_txt(mdl_ref_file)
        print("\tFinished loading data")

        # maps from ref order idx to mdl order idx
        if "src" in mdl_ref_file: # compare against sources
            mdl_idxs = get_aligned_shortest(sorted_ref_srcs, mdl_refs, n_exs_to_search)
        else: # compare against gold targets / references
            assert "ref" in mdl_ref_file
            mdl_idxs = get_aligned_shortest(sorted_ref_trgs, mdl_refs, n_exs_to_search)
        print("\tFinished aligning data")
        subset_mdl_gens = [mdl_gens[i] for i in mdl_idxs]
        write_txt(subset_mdl_gens, f"data/subset{n_exs}.{mdl_name}.trg.ref_order.txt")
        print("\tFinished writing data")

    return


def prepare_parlai_data():
    """ Prepare data for ParlAI mturk tasks """

    # load original articles
    mdl_files = {
                 #'src': 'data/subset-src.txt',
                 #'bus': 'data/subset-bus.txt',
                 #'fas': 'data/subset-fas.txt',
                 #'pgc': 'data/subset-pgc.txt',
                 #'trg': 'data/subset-trg.txt'

                 'src': 'data/subset500.src.txt',
                 'trg': 'data/subset500.trg.txt',
                 'bus': 'data/subset500.bus.trg.ref_order.txt',
                 'fas': 'data/subset500.fas.trg.ref_order.txt',
                 'pgc': 'data/subset500.pgc.trg.ref_order.txt',
                }

    raw_srcs = [s.strip() for s in open(mdl_files['src'], encoding="utf-8")]
    srcs = []
    for src in raw_srcs:
        for k, v in JOIN_PUNCT.items():
            src = src.replace(k, v)
        srcs.append(src)
    srcs_sents = [[detokenize_sent(s) for s in sent_tokenize(src)] for src in srcs]

    n_shards = 5
    should_add_attn_task = 1
    should_filter_length = 0
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

            gen_sents = [detokenize_sent(s) for s in sent_tokenize(gen)]
            for sent_idx, sent in enumerate(gen_sents):
                gen_d = {'dialog': [{'speaker': 'model', 'text': sent}],
                         'ex_idx': (mdl_name, src_idx, sent_idx),
                         'para_idx': sent_idx
                        }
                par_sents.append(gen_d)

            if should_add_attn_task:
                use_negative = bool(random.random() > 0.5)
                if use_negative:
                    # negative attn task
                    sent = random.choice(gen_sents).split()
                    random.shuffle(sent)
                    sent = ' '.join(sent).lower().capitalize()
                else:
                    # positive attn task
                    #sent = src_sents[min(len(src_sents), -2)]
                    sent = src_sents[min(len(src_sents), 1)]

                sent_idx = -2
                gen_d = {'dialog': [{'speaker': 'model', 'text': sent}],
                         'ex_idx': (mdl_name, src_idx, sent_idx),
                         'para_idx': sent_idx,
                         'answer': 'no' if use_negative else 'yes'
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
            para_file = f"data/mturk/summary/{mdl_name}_para_nex500_ref_order_shard{shard_n}.jsonl"
            with open(para_file, 'w', encoding='utf-8') as para_fh:
                data_slice = para_data[shard_n * n_exs_per_shard : (shard_n + 1) * n_exs_per_shard]
                for para_d in data_slice:
                    para_fh.write(f"{json.dumps(para_d)}\n")

            # write to jsonl
            sent_file = f"data/mturk/summary/{mdl_name}_sent_nex500_ref_order_shard{shard_n}.jsonl"
            with open(sent_file, 'w', encoding='utf-8') as sent_fh:
                data_slice = sent_data[shard_n * n_exs_per_shard : (shard_n + 1) * n_exs_per_shard]
                for sents_d in data_slice:
                    for sent_d in sents_d:
                        sent_fh.write(f"{json.dumps(sent_d)}\n")



def evaluate_parlai_mturk(data_files, mdl):
    """ Do basic data analysis on mturk data (run through ParlAI)
    data_file should be jsonl
    """

    idx2responses = defaultdict(lambda: defaultdict(list))
    idx2data = dict()
    worker2responses = defaultdict(list)
    response_map = {'1': 'yes', '2': 'no'}
    n_resps = 0

    for data_file in data_files:
        data = [ast.literal_eval(l) for l in open(data_file, encoding="utf-8")]
        for datum in data:
            for worker_id, worker in datum['worker_data'].items():
                if worker['response']['text'] in MTURK_BAD_RESPONSES:
                    continue
                short_msg_flag, attn_fail_flag = 0, 0
                ## filter out short responses
                for response in worker['response']['task_data']:
                    if not response.get('textReason', ''):
                        short_msg_flag = True
                ## filter out attn check fails
                for task_idx, task in enumerate(worker['task_data']):
                    if task['conversations'][1].get('answer', None) is not None:
                        choice = int(worker['response']['task_data'][task_idx]['speakerChoice'])
                        expected = 1 if task['conversations'][1]['answer'] == 'yes' else 2
                        attn_fail_flag = bool(choice != expected)
                if short_msg_flag or attn_fail_flag:
                    continue

                # bookkeeping
                n_resps += 1
                worker2responses[worker_id].append(worker)

                para_idx = tuple(worker['task_data'][0]['conversations'][0]['ex_idx'])
                idx2data[para_idx] = worker['task_data'][0]['conversations'][0]['dialog'][0]['text']

                sent_idxs = []
                for task in worker['task_data']:
                    sent_idx = tuple(task['conversations'][1]['ex_idx'])
                    sent_idxs.append(sent_idx)
                    idx2data[sent_idx] = task['conversations'][1]['dialog'][0]['text']

                resps = [d["speakerChoice"] for d in worker['response']['task_data']]

                for sent_idx, resp in zip(sent_idxs, resps):
                    idx2responses[para_idx][sent_idx].append(resp)


    def print_ids():
        n_hits, n_tasks = 0, 0
        n_yes, n_no = 0, 0
        n_all_yes, n_all_no = 0, 0
        multiresponse_idxs = []
        all_no_idxs = []
        resps3_idxs = []
        for k, v in idx2responses.items():
            n_responses = max(len(vv) for vv in v.values())
            if n_responses == 3:
                resps3_idxs.append(k)
            resps = [1 if vv[0] == '1' else 0 for vv in v.values()]
            n_all_yes += int(sum(resps) == len(v))
            n_all_no += int(sum(resps) == 0)
            n_yes += sum(resps)
            n_no += len(v) - sum(resps)
            n_tasks += len(v)
            n_hits += n_responses

            if n_responses > 1:
                multiresponse_idxs.append(k)
            if not sum(resps):
                all_no_idxs.append(k)
            print(f"{k[1]}: {n_responses} responses")

        print(f"loaded data from {n_hits} hits for {mdl}")
        print(f"# examples w/ >1 responses: {len(multiresponse_idxs)}")
        print(f"# examples w/ >3 responses: {len(resps3_idxs)}")
        print(f"\tids: {multiresponse_idxs}")

        assert n_tasks == (n_yes + n_no)
        print(f"\tn yes: {n_yes}, n no: {n_no}, n all yes {n_all_yes}, n all no: {n_all_no}")
        print(f"\tall no idxs: {all_no_idxs}")
        return resps3_idxs

    def print_response(par_idx):
        src_key = ('src', par_idx, -1)
        n_trgs = len(idx2responses[src_key])
        print(f"Paragraph {par_idx}")
        print(f"\t{idx2data[src_key]}")
        #for i in range(n_trgs):
        for trg_key in idx2responses[src_key].keys():
            #trg_key = (mdl, par_idx, i)
            print(f"\t{idx2responses[src_key][trg_key]}: {idx2data[trg_key]}")

    resps3_idxs = print_ids()
    ipdb.set_trace()



def compute_correctness_judgments_rouge_correlations(turk_files, hyp_file, mdl, ref_file='data/subset-trg.txt'):
    """ Compute sentence and system level correlations
    between human annotations and ROUGE scores
    """

    # 1 is YES, 2 is NO
    resp_map = {'1': 1, '2': 0}

    # Load mturk data
    n_hits = 0
    idxs = []
    idx2responses = defaultdict(lambda: defaultdict(list))
    for turk_file in turk_files:
        mturk_data = [ast.literal_eval(l) for l in open(turk_file, encoding="utf-8")]
        for datum in mturk_data:
            for worker_id, worker in datum['worker_data'].items():

                # Filter out bad reponses
                ## filter out returns and discounnects
                if worker['response']['text'] in MTURK_BAD_RESPONSES:
                    continue
                short_msg_flag, attn_fail_flag = 0, 0
                ## filter out short responses
                for response in worker['response']['task_data']:
                    if not response.get('textReason', ''):
                        short_msg_flag = True
                ## filter out attn check fails
                for task_idx, task in enumerate(worker['task_data']):
                    if task['conversations'][1].get('answer', None) is not None:
                        choice = int(worker['response']['task_data'][task_idx]['speakerChoice'])
                        expected = 1 if task['conversations'][1]['answer'] == 'yes' else 2
                        attn_fail_flag = bool(choice != expected)
                if short_msg_flag or attn_fail_flag:
                    continue
                n_hits += 1

                para_idx = tuple(worker['task_data'][0]['conversations'][0]['ex_idx'])[1]
                idxs.append(para_idx)
                sent_idxs = [t['conversations'][1]['ex_idx'][2] for t in worker['task_data']]
                resps = [d["speakerChoice"] for d in worker['response']['task_data']]
                for sent_idx, resp in zip(sent_idxs, resps):
                    idx2responses[para_idx][sent_idx].append(resp)
    idxs = list(set(idxs))

    # Aggregate stuff
    n_tasks = 0
    human_scores = list()
    odd_human_scores = list()
    odd_idxs = list()
    disagreement_idxs = list()
    equal_idxs = list()
    for para_idx in idxs:
        para_d = idx2responses[para_idx]
        agg_labels = []
        odd_agg_labels = []
        for sent_idx, votes in para_d.items():
            if sent_idx == ATTN_IDX:
                continue
            votes = [resp_map[v] for v in votes]
            votes0 = votes.count(0)
            votes1 = votes.count(1)
            agg_labels.append(1 if votes1 >= votes0 else 0)
            if votes0 > 0 and votes1 > 0:
                disagreement_idxs.append((para_idx, sent_idx))
            if votes0 == votes1:
                equal_idxs.append((para_idx, sent_idx))
            if len(votes) % 2 == 1:
                odd_agg_labels.append(1 if votes1 > votes0 else 0)
                if para_idx not in odd_idxs:
                    odd_idxs.append(para_idx)
            n_tasks += 1

        human_scores.append(sum(agg_labels) / len(agg_labels))
        if odd_agg_labels:
            odd_human_scores.append(sum(odd_agg_labels) / len(odd_agg_labels))

    print(f"Loaded data from {len(idxs)} examples, {n_hits} tasks")
    print(f"\t{len(odd_human_scores)} / {len(human_scores)} examples with odd number of labels")
    print(f"\t{len(disagreement_idxs)} / {n_tasks} tasks with disagreement")
    print()

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

        n_qsts_per_doc = 10
        qags_src_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset/prd.qst{n_qsts_per_doc}-gen.cnndm-src.json"
        qags_trg_file = f"/misc/vlgscratch4/BowmanGroup/awang/ckpts/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/{mdl}-subset/prd.qst{n_qsts_per_doc}-gen.cnndm-gen.json"
        all_qags_scores = get_qags_scores(qags_src_file, qags_trg_file, metric_name)
        qags_scores = [all_qags_scores[idx] for idx in idxs]
        pearson_corr = pearsonr(scores, qags_scores)
        spearman_corr = spearmanr(scores, qags_scores)
        print(f"pearson correlation w/ QAGS {metric_name}: {pearson_corr}")
        print(f"spearman correlation w/ QAGS {metric_name}: {spearman_corr}")


    print(f"All examples")
    compute_rouge_correlation(idxs, human_scores)
    compute_qags_correlation(idxs, human_scores, metric_name="em")
    compute_qags_correlation(idxs, human_scores, metric_name="f1")
    print()

    print(f"Examples with odd # labels")
    compute_rouge_correlation(odd_idxs, odd_human_scores)
    compute_qags_correlation(odd_idxs, odd_human_scores, metric_name="em")
    compute_qags_correlation(odd_idxs, odd_human_scores, metric_name="f1")



def compute_pair_judgments_rouge_correlations():
    """ Compute sentence and system level correlations
    between human annotations and ROUGE scores
    """

    model = "bus"

    data_file = "data/mturk_summary_pair.csv"
    all_data = pd.read_csv(data_file)

    ref_file = 'data/subset-trg.txt'
    all_refs = [l.strip() for l in open(ref_file, encoding='utf-8')]
    refs = []
    hyps = []
    choices = []
    for idx, row in all_data.iterrows():
        refs.append(all_refs[row['Input.idx']])
        refs.append(all_refs[row['Input.idx']])
        hyps.append(row['Input.summary1'])
        hyps.append(row['Input.summary2'])
        if row['Answer.choice.summary1']:
            choices.append(1)
        elif row['Answer.choice.summary2']:
            choices.append(2)
        elif row['Answer.choice.about-equal']:
            choices.append(3)
        else: # 'Answer.choice.cant-tell'
            choices.append(4)

    assert len(refs) == len(hyps)

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

    rouge_scores = rouge_eval.get_scores(hyps, refs)
    def score(score_d, idx):
        total_score = 0.
        n_metrics = len(score_d)
        for metric_name, metric_d in score_d.items():
            total_score += metric_d[idx]['f'][0]
        return total_score / n_metrics


    n_agree = 0
    n_canttell = 0
    rouge_choices = []
    scores1, scores2 = [], []
    for i in range(int(len(refs) / 2)):
        score1 = score(rouge_scores, 2*i)
        score2 = score(rouge_scores, 2*i + 1)
        scores1.append(score1)
        scores2.append(score2)

        if (abs(score1 - score2) < 0.01):
            rouge_choices.append(3)
        elif score1 > score2:
            rouge_choices.append(1)
        elif score1 < score2:
            rouge_choices.append(2)

        if (abs(score1 - score2) < 0.01 and choices[i] == 3) or \
           (score1 > score2 and choices[i] == 1) or \
           (score1 < score2 and choices[i] == 2):
            n_agree += 1
        if choices[i] == 4:
            n_canttell += 1

    print(f"ROUGE and human eval agree {n_agree}/{len(choices)} ({n_agree / len(choices)} %)")
    print(f"\t# cant tell (human): {n_canttell}")

    ipdb.set_trace()



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



def mturk_review_hits(is_sandbox=False):
    """Currently: analyze time
    """
    data_handler = MTurkDataHandler(file_name='pmt_sbdata.db' if is_sandbox else 'pmt_data.db')
    worker_hit_pairs = [l.strip().split(',') for l in open("workers_to_check.txt", encoding="utf-8")][1:]

    for worker_id, hit_id in worker_hit_pairs:
        ipdb.set_trace()
        asgs = data_handler.get_assignments_for_run(run['run_id'])
        for asg in asgs:
            asg_data = data_handler.get_worker_assignment_pairing(asg['worker_id'], asg['assignment_id'])
            if asg_data['status'] in ['disconnect']:
                continue
            if asg_data['task_end'] is None or asg_data['task_start'] is None:
                continue
            statuses.append(asg_data['status'])
            times.append(asg_data['task_end'] - asg_data['task_start'])



mdl2turk_data = {
    #"bus": ["data/mturk/summary/precision/mturk_data.09271534.jsonl",
    #        "data/mturk/summary/precision/mturk_data.10041456.jsonl",
    #        "data/mturk/summary/precision/mturk_data.10071418.jsonl"],
    "bus": ["data/mturk/summary/precision/mturk_data.10111337.jsonl"],

    "trg": ["data/mturk/summary/precision/mturk_data.09271635.jsonl"],

    #"pgc": "data/mturk/summary/precision/mturk_data.09271736.jsonl",
    #"pgc": "data/mturk/summary/precision/mturk_data.10021638.jsonl",
    #"pgc": "data/mturk/summary/precision/mturk_data.10031605.jsonl",
    "pgc": [
            "data/mturk/summary/precision/mturk_data.09271736.jsonl",
            "data/mturk/summary/precision/mturk_data.10021638.jsonl",
            "data/mturk/summary/precision/mturk_data.10031605.jsonl"
           ],

    #"fas": ["data/mturk/summary/precision/mturk_data.10011138.jsonl"]
    #"fas": "data/mturk/summary/precision/mturk_data.10021758.jsonl"
    "fas": ["data/mturk/summary/precision/mturk_data.10011138.jsonl",
            "data/mturk/summary/precision/mturk_data.10021758.jsonl",
            "data/mturk/summary/precision/mturk_data.10071607.jsonl",
           ]
}
mdl = "bus"

#extract_src_trg_gen_from_fseq_log()
#extract_questions_and_write_jsonl()
#aggregate_questions()
#format_abstractive_qa()
#process_human_subset()
#compute_pair_judgments_rouge_correlations()

#align_summaries()
#prepare_parlai_data()

evaluate_parlai_mturk(mdl2turk_data[mdl], mdl)

#compute_correctness_judgments_rouge_correlations(turk_files=mdl2turk_data[mdl],
#                                                 hyp_file=f"data/subset-{mdl}.txt",
#                                                 mdl=mdl)

#mturk_posthoc()
#mturk_review_hits()
