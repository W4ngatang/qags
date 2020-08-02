""" Use a BERT tokenizer to tokenize all files in a directory """
import logging as log
log.basicConfig(format="%(asctime)s: %(message)s",
                datefmt="%m/%d %I:%M:%S %p",
                level=log.INFO)
import os
import sys
import argparse

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

def tokenize(data_file, out_file, tokenizer):
    with open(data_file, "r", encoding="utf-8") as data_fh:
        data = data_fh.readlines()

    tok_data = []
    for datum in data:
        tok_data.append(tokenizer.tokenize(datum))

    with open(out_file, "w", encoding="utf-8") as out_fh:
        for datum in tok_data:
            out_fh.write(f"{' '.join(datum)}\n")


def main(arguments):
    parser = argparse.ArgumentParser(description="Mostly just tokenization")
    parser.add_argument("--bert-version", help="version of BERT tokenizer to use",
                        default="bert-base-uncased")
    parser.add_argument("--data-dir", help="directory containing data to tokenize")
    parser.add_argument("--out-dir", help="directory to write tokenized data")
    args = parser.parse_args(arguments)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(args.bert_version)

    files = os.listdir(args.data_dir)
    for data_file in files:
        full_data_file = os.path.join(args.data_dir, data_file)
        if os.path.isdir(full_data_file):
            continue
        log.info(f"Tokenizing {data_file}...")
        file_parts = data_file.split(".")
        file_prefix = ".".join(file_parts[:-1])
        file_suffix = file_parts[-1]
        out_file = os.path.join(args.out_dir, f"{file_prefix}.tok.{file_suffix}")
        tokenize(full_data_file, out_file, tokenizer)
        log.info(f"\tDone! Wrote tokenized file to {out_file}")

if __name__ == "__main__":
    main(sys.argv[1:])
