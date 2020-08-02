#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import functools
import sys

from collections import Counter
from multiprocessing import Pool

from gpt2_encoding import get_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-json", required=True)
    parser.add_argument("--vocab-bpe", required=True)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--inputs", nargs="+", default=['-'], help="input files to filter/encode")
    parser.add_argument("--outputs", nargs="+", default=['-'], help="path to save encoded outputs")
    parser.add_argument("--min-len", type=int, metavar="N", help="filter sentence pairs with fewer than N tokens")
    parser.add_argument("--max-len", type=int, metavar="N", help="filter sentence pairs with more than N tokens")
    parser.add_argument("--keep-empty", action="store_true", help="keep empty lines")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
            "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8")) \
                if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8")) \
                if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        if args.decode:
            encoded_lines = pool.imap(encoder.decode_lines, zip(*inputs), 100)
        else:
            encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def valid(self, tokens):
        if self.args.min_len is not None or self.args.max_len is not None:
            return (
                (self.args.min_len is None or len(tokens) >= self.args.min_len)
                and (self.args.max_len is None or len(tokens) <= self.args.max_len)
            )
        else:
            return True

    def encode_lines(self, lines):
        """Encode a set of lines. All lines will be encoded (or filtered) together."""
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            if not self.valid(tokens):
                return ["FILTERED", None]
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            new_tokens = self.decode(tokens)
            dec_lines.append(new_tokens)
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
