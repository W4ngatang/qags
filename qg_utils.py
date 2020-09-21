"""  """
import os
import sys
import json
import random
import argparse
from collections import defaultdict, Counter

import spacy
from transformers import GPT2Tokenizer

from utils import write_data, write_jsonl, write_txt, \
                  process, print_samples, format_squad, \
                  filter_line_fseq, parse_generation, \
                  load_txt, load_json


ANS_TOK = "[ANS]"
NO_ANS_TOK = "[NO_ANS]"


def extract_ans(txts):
    """ extract entities from a sentence using spacy

    rules:
        - entities (non-pronoun)
            - each portion of a person's name
        - noun chunks (non-pronoun)
            - adjectives within noun chunks
            - nouns w/ dependencies that are proper nouns, roughly nouns modifying proper nouns
            - if the head of a noun chunk if a verb, the entire noun chunk ?
    """
    nlp = get_spacy_nlp("en_core_web_lg")
    all_ans = list()
    for doc in nlp.pipe(txts, disable=[]):
        ans = list()
        for ent in doc.ents:
            ans.append(ent.text)
        for chunk in doc.noun_chunks:
            ans.append(chunk.text)
        ans = list(set(ans))
        all_ans.append(ans)
    return all_ans


def prepare_ans_conditional_data(data_file,
                                 out_dir,
                                 out_prefix,
                                 n_ans_per_txt=10,
                                 use_no_ans=False,
                                 use_only_no_ans=False,
                                 ):
    """ Given a text file, extract possible answer candidates for each line.

    Will generate n_ans_per_text instances for each line in txt
    """


    txt_w_ans_file = f"{out_dir}/{out_prefix}_w_{n_ans_per_txt}ans.txt"
    txt_file = f"{out_dir}/{out_prefix}.txt"
    ans_file = f"{out_dir}/{out_prefix}_{n_ans_per_txt}ans.txt"

    print(f"Preparing answer conditional question generation data for {data_file}")
    if use_only_no_ans:
        print("\twith ONLY NO_ANS!")
    elif use_no_ans:
        print("\twith NO_ANS option!")
    else:
        print("\twithout NO_ANS option!")

    all_txts = load_txt(data_file)
    print("Extracting entities...")
    all_anss = extract_ans(all_txts)
    print("\tDone!")
    print(f"\tMin ans count: {min(len(a) for a in all_anss)}")
    print(f"\tMax ans count: {max(len(a) for a in all_anss)}")

    print("Writing...")
    txts_w_ans = list()
    all_txt = list()
    all_ans = list()
    for txt, anss in zip(all_txts, all_anss):
        if use_only_no_ans:
            anss = [NO_ANS_TOK] * n_ans_per_txt
        elif use_no_ans:
            if len(anss) > n_ans_per_txt - 1:
                anss = random.sample(anss, k=n_ans_per_txt - 1)
            anss += [NO_ANS_TOK] * (n_ans_per_txt - len(anss))
            assert NO_ANS_TOK in anss, ipdb.set_trace()
        else:
            if len(anss) < n_ans_per_txt:
                extra_anss = random.choices(anss, k=n_ans_per_txt - len(anss))
                anss += extra_anss
            if len(anss) > n_ans_per_txt:
                anss = random.sample(anss, n_ans_per_txt)
            assert len(anss) == n_ans_per_txt, ipdb.set_trace()

        for ans in anss:
            txts_w_ans.append(f"{txt} {ANS_TOK} {ans}")
            all_txt.append(txt)
            all_ans.append(ans)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(txt_w_ans_file, 'w') as out_fh:
        for txt in txts_w_ans:
            out_fh.write(f'{txt}\n')
    with open(txt_file, 'w') as out_fh:
        for txt in all_txt:
            out_fh.write(f'{txt}\n')
    with open(ans_file, 'w') as out_fh:
        for ans in all_ans:
            out_fh.write(f'{ans}\n')
    print("\tDone!")
    print(f"\tWrote {len(txts_w_ans)} sentences to {txt_w_ans_file}")


def extract_gen_from_fseq_log(data_file, out_dir):
    """ """

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    data = parse_generation(data_file)

    n_gens = 0
    gen_fh = open(f'{out_dir}/gens.txt', 'w')
    prob_fh = open(f'{out_dir}/probs.txt', 'w')
    ex_ids = sorted(list(data.keys()))
    for ex_id in ex_ids:
        ex_gens = data[ex_id]['gen']
        for raw, prob in ex_gens:
            tok_str = raw.replace('<s>', '').replace('<mask>', '').strip().split()
            tok_ids = [int(t) for t in tok_str]
            gen = tokenizer.decode(tok_ids)
            gen_fh.write(f'{gen}\n')
            prob_fh.write(f'{prob}\n')
            n_gens += 1

    print(f'Wrote {n_gens} generations to {out_dir}')


def main(arguments):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--command", choices=["extract_ans", "extract_gen"], help="Function to perform")
    parser.add_argument("--data_file", type=str, help="File from which to extract answers or filter questions. For `extract_ans`, this should be a text file with an example per line.")
    parser.add_argument("--out_dir", type=str, help="Directory to write outputs")
    parser.add_argument("--out_prefix", type=str, default="test", help="Prefix for files written out")

    # answer extraction options
    parser.add_argument("--n_ans", type=int, default=10, help="Number of answer candidates per example")

    args = parser.parse_args(arguments)

    if args.command == "extract_ans":
        prepare_ans_conditional_data(args.data_file, args.out_dir, args.out_prefix,
                                     n_ans_per_txt=args.n_ans)
    elif args.command == "extract_gen":
        extract_gen_from_fseq_log(args.data_file, args.out_dir)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
