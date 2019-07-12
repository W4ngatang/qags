""" Do various analysis things """
import json
import random

N_SAMPLE = 5

src_qst_file = "data/questions.cnndm-sources.sources.json"
trg_qst_file = "data/questions.cnndm-sources.targets.json"
src_ans_file = "/checkpoint/wangalexc/ppb/bert-large-uncased/squad_v2_0/06-25-2019-v1-1/predictions.cnndm-sources.sources.json"
trg_ans_file = "/checkpoint/wangalexc/ppb/bert-large-uncased/squad_v2_0/06-25-2019-v1-1/predictions.cnndm-sources.targets.json"

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

print("***** Matching answers *****")
for idx in sample_match:
    qst_idx = int(idx)
    qst = src_qsts[qst_idx]['paragraphs'][0]['qas'][0]['question']
    src = src_qsts[qst_idx]['paragraphs'][0]['context']
    trg = trg_qsts[qst_idx]['paragraphs'][0]['context']
    print(f"Question: {qst}")
    print(f"Answer: {src_anss[idx]}")
    print(f"Source text: {src}")
    print(f"Target text: {trg}")
    print()

print("***** Non-matching answers *****")
for idx in sample_nonmatch:
    qst_idx = int(idx)
    qst = src_qsts[qst_idx]['paragraphs'][0]['qas'][0]['question']
    src = src_qsts[qst_idx]['paragraphs'][0]['context']
    trg = trg_qsts[qst_idx]['paragraphs'][0]['context']
    print(f"Question: {qst}")
    print(f"Answer using source: {src_anss[idx]}")
    print(f"Answer using target: {trg_anss[idx]}")
    print()
    print(f"Source text: {src}\n")
    print(f"Target text: {trg}\n")
