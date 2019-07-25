# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetune unidirectional LMs for conditional sequence generation. """

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S', level=log.INFO)
import os
import sys
import csv
import json
import time
import shutil
import random
import argparse

from tqdm import tqdm, trange

import spacy
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tensorboardX import SummaryWriter
from pytorch_transformers import OpenAIGPTDoubleHeadsModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, \
                                 AdamW, WEIGHTS_NAME, CONFIG_NAME, \
                                 GPT2DoubleHeadsModel, GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers.optimization import WarmupLinearSchedule


MC_TASK_NAMES = {'rocstories'}


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def load_data_and_tokenize(data_dir, split, task_name, special2toks):
    """ Load data from files and return as list of examples

    args:

    returns:
        - examples (List[Tuple()]): list of examples represented as tuples,
            contents of which vary based on the task,
            e.g. for ROC Stories, output a list of story, 1st continuation, 2nd continuation, label
    """

    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    def format_text(text):
        """Standardizes text using OpenAI GPT's tokenizer (includes lowercasing)"""
        return tokenizer.decode(tokenizer.encode(text.strip())).strip()

    assert split in {'train', 'dev'}, 'Split "{}" not yet supported'.format(split)
    examples = []  # Fill examples based on task_name

    if task_name == 'rocstories':
        file_version = 'test' if split == 'dev' else 'val'
        dataset_path = '{0}/rocstories/cloze_test_{1}__spring2016 - cloze_test_ALL_{1}.csv'.format(
            DATA_DIR, file_version)
        with open(dataset_path, encoding='utf_8') as f:
            f = csv.reader(f)
            next(f)  # Skip the first line
            for line in tqdm(f):
                examples.append((' '.join(line[1:5]), line[5], line[6], int(line[-1])-1))

    elif task_name == "squad-freetext":
        with open(f'{data_dir}/{split}.json', encoding="utf-8") as f:
            data = json.load(f)["data"]

        for doc in data:
            for psg_d in doc["paragraphs"]:
                psg = psg_d["context"]
                for qst_d in psg_d["qas"]:
                    qst = qst_d["question"]
                    anss = [a["text"] for a in qst_d["answers"]]
                    if len(anss) == 0: # question has no answer
                        ans = special2toks['no_ans_tok']
                    else:
                        ans = anss[0]

                    examples.append((format_text(qst), format_text(psg), format_text(ans)))

    else:
        raise NotImplementedError(task_name)

    log.info(f'Read {len(examples)} examples.')
    assert len(examples) > 0, 'Error: Read 0 examples.'
    return examples


def index(tokenizer, obj):
    """ Recursively index all objects in an interable using tokenizer """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    elif isinstance(obj, int):
        return obj
    return list(index(tokenizer, o) for o in obj)


def tensorfy(data, max_seq_len, task_name, special2idx,
             no_input_lm=False, eoi_idx=None):
    """ Convert indexed dataset into tensors

    args:
        - data (List[Tuple()]): data split, where each split is
            list of examples represented as tuples, contents depend on task
            - [ROCStories] (story, 1st continuation, 2nd continuation, label)
            - [Language Modeling] (sequence of tokens)
        - max_seq_len (int): maximum sequence length of the model
        - special2idx (Dict[str: int]): dict containing mappings from
            special tokens (to be used by each task) to ints
        - no_input_lm (Bool): if True, don't use labels for the inputs, must also give eoi_idx
        - eoi_idx (Int): end of input idx; must be provided if `no_input_lm`

    returns:
        - tsr_data (List[Tuple(Tensor)]): dict of data splits,
            where a split is tuples of Torch tensors

    """

    n_batch = len(data)
    if task_name == 'rocstories':
        sos_idx = special2idx['sos_tok']
        delim_idx = special2idx['delim_tok']
        clf_idx = special2idx['clf_tok']
        cap_len = max_seq_len // 2 - 2
        input_len = max(len(story[:cap_len]) + max(len(cont1[:cap_len]), len(cont2[:cap_len])) + 3
                        for story, cont1, cont2, _ in data)
        input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
        lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
        mc_labels = np.zeros((n_batch,), dtype=np.int64)
        for i, (story, cont1, cont2, mc_label), in enumerate(data):
            with_cont1 = [sos_idx] + story[:cap_len] + [delim_idx] + cont1[:cap_len] + [clf_idx]
            with_cont2 = [sos_idx] + story[:cap_len] + [delim_idx] + cont2[:cap_len] + [clf_idx]
            input_ids[i, 0, :len(with_cont1)] = with_cont1
            input_ids[i, 1, :len(with_cont2)] = with_cont2
            mc_token_ids[i, 0] = len(with_cont1) - 1
            mc_token_ids[i, 1] = len(with_cont2) - 1
            lm_labels[i, 0, :len(with_cont1)] = with_cont1
            lm_labels[i, 1, :len(with_cont2)] = with_cont2
            mc_labels[i] = mc_label
        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)

    elif task_name == 'squad-freetext':
        sos_idx = special2idx['sos_tok']
        delim_idx = special2idx['delim_tok']
        max_ans_len = max(len(ans) for _, _, ans in data)
        cap_len = (max_seq_len - max_ans_len) // 2
        input_len = max(len(qst[:cap_len]) + len(psg[:cap_len]) + len(ans) + 4 for qst, psg, ans in data)
        input_ids = np.zeros((n_batch, 1, input_len), dtype=np.int64)
        lm_labels = np.full((n_batch, 1, input_len), fill_value=-1, dtype=np.int64)
        for i, (qst, psg, ans) in enumerate(data):
            ex = [sos_idx] + qst[:cap_len] + [delim_idx] + psg[:cap_len] + [eoi_idx] + ans + [eoi_idx]
            input_ids[i, 0, :len(ex)] = ex
            if no_input_lm:
                assert eoi_idx is not None and eoi_idx in ex, f'end of input token {eoi_tok} not in example {ex}'
                first_output_index = ex.index(eoi_idx) + 1
                lm_labels[i, 0, first_output_index: len(ex)] = ex[first_output_index: len(ex)]
            else:
                lm_labels[i, 0, :len(ex)] = ex
        all_inputs = (input_ids, lm_labels)

    else:
        input_ids = np.zeros((n_batch, 1, input_len), dtype=np.int64)
        lm_labels = np.full((n_batch, 1, input_len), fill_value=-1, dtype=np.int64)
        for i, seq, in enumerate(data):
            input_ids[i, 0, :len(seq)] = seq
            if no_input_lm:
                assert eoi_idx is not None and eoi_idx in seq, f'end of input token {eoi_idx} not in example {seq}'
                first_output_index = seq.index(eoi_idx) + 1
                lm_labels[i, 0, first_output_index: len(seq)] = seq[first_output_index: len(seq)]
            else:
                lm_labels[i, 0, :len(seq)] = seq
        all_inputs = (input_ids, lm_labels)

    return tuple(torch.tensor(t) for t in all_inputs)


def create_loader(data, batch_size):
    """ """
    dataset = TensorDataset(*data)
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return data_loader


def get_model_tokenizer(model_name, special_tokens):
    """ Returns a tokenizer's Python class """
    if model_name == 'openai-gpt':
        tokenizer_cls = OpenAIGPTTokenizer
    elif model_name in ['gpt2', 'gpt2-medium']:
        tokenizer_cls = GPT2Tokenizer
    else:
        raise ValueError(f"Invalid model class {model_name}")
    tokenizer = tokenizer_cls.from_pretrained(model_name)
    if model_name in ['gpt2', 'gpt2-medium']: # HACK
        tokenizer.unk_token = tokenizer.eos_token
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def get_pretrained_model(model_name, task_name, tokenizer):
    """ Returns a model's Python class """
    if task_name in MC_TASK_NAMES:
        if model_name == 'openai-gpt':
            mdl_cls = OpenAIGPTDoubleHeadsModel
        else:
            mdl_cls = GPT2DoubleHeadsModel
    else:
        if model_name == 'openai-gpt':
            mdl_cls = OpenAIGPTLMHeadModel
        else:
            mdl_cls = GPT2LMHeadModel
    model = mdl_cls.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    return model


def get_optimizer_and_scheduler(model, data, args):
    """ Lazy abstraction around build the optimizer and LR scheduler """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(data) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          #max_grad_norm=args.max_grad_norm,
                          weight_decay=args.weight_decay#,
                          #t_total=num_train_optimization_steps
                          )
    scheduler = WarmupLinearSchedule(optimizer=optimizer,
                                     warmup_steps=args.warmup_proportion * num_train_optimization_steps,
                                     t_total=num_train_optimization_steps)

    return optimizer, scheduler


def load_model(saved_dir):
    """ Loads a previously saved model """
    output_args_file = os.path.join(saved_dir, 'training_args.bin')
    args = torch.load(output_args_file)
    print('Loaded args:', args)
    tokenizer_class = get_tokenizer_class(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(saved_dir)
    model_class = get_model_class(args.model_name, args.task_name)
    model = model_class.from_pretrained(saved_dir)
    return model, tokenizer, args


def save_model(model, tokenizer, args, out_dir, weights_name=WEIGHTS_NAME, override_default_weights=True):
    """ Saves and existing model """
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join( out_dir, weights_name)
    output_default_model_file = os.path.join(out_dir, WEIGHTS_NAME)
    output_config_file = os.path.join( out_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    if override_default_weights:
        torch.save(model_to_save.state_dict(), output_default_model_file)

    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.out_dir)

    output_args_file = os.path.join(out_dir, 'training_args.bin')
    torch.save(args, output_args_file)
    return


def setup(outdir, seed, overwrite_output_dir):
    """ Setup """

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #elif overwrite_output_dir:
    #    log.info(f'Overwriting existing output directory {outdir}')
    #    shutil.rmtree(outdir)
    #    os.makedirs(outdir)

    # Log file
    log_file = os.path.join(outdir, "train.log")
    log_fh = log.FileHandler(log_file)
    log_fmt = log.Formatter("%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p")
    log_fh.setFormatter(log_fmt)
    log.getLogger().addHandler(log_fh)

def main(arguments):
    parser = argparse.ArgumentParser()

    # logistics
    parser.add_argument('--data_dir', default='~/data', type=str, help='directory containing data')
    parser.add_argument('--out_dir', type=str, help='directory to write outputs')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42)

    # Preprocessing options
    parser.add_argument('--reload_data', action='store_true')

    #
    parser.add_argument('--model_name', default='gpt2', type=str, choices=['gpt2', 'gpt2-medium', 'openai-gpt'],
                        help='model name')
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train.",
                        choices=['rocstories', 'squad.sf-q', 'squad-freetext'])

    # Training details
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--no_input_lm_train', action='store_true', help="Use LM loss on input while training?")
    parser.add_argument('--no_input_lm_eval', action='store_true', help="Use LM loss on input while evaluating?")

    # CUDA / GPU stuff
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args(arguments)
    print(args)

    # Logging and outputs
    setup(outdir=args.out_dir, seed=args.seed, overwrite_output_dir=args.overwrite_output_dir)

    # GPU stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    log.info("device: {}, n_gpu {}, 16-bits training: {}".format(device, n_gpu, args.fp16))
    assert args.gradient_accumulation_steps >= 1, f"gradient_accumulation_steps should be >= 1, found {args.gradient_accumulation_steps}"
    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    eval_batch_size = 2 * args.train_batch_size
    task_name = args.task_name

    # Load model tokenizer (since pretrained models have a particular vocabulary)
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be learned
    if args.task_name in MC_TASK_NAMES:
        special_tokens = {'sos_tok': '_start_',
                          'delim_tok': '_delimiter_',
                          'clf_tok': '_classify_'
                         }
    elif args.task_name == "squad-freetext":
        special_tokens = {'sos_tok': '_start_',
                          'delim_tok': '_delimiter_',
                          'eos_tok': '_answer_',
                          'no_ans_tok' :'__no_ans__'
                         }
    else:
        special_tokens = []
    tokenizer = get_model_tokenizer(args.model_name, special_tokens)
    special_tokens_ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in special_tokens.items()}

    # Load pretrained model
    model = get_pretrained_model(args.model_name, task_name, tokenizer)
    if args.fp16:
        model.half()
    model.to(device)


    # Preprocess or load cached preprocessed data by
    # 1) load data and tokenize
    # 2) index the data
    # 3) tensorfy the data
    # 4) create data loaders
    splits = ["train", "dev"]
    split2data = {}
    for split in splits:
        cached_data_file = f'{args.out_dir}/{args.task_name}.indexed_data.{split}.json'
        if os.path.exists(cached_data_file) and not args.reload_data:
            log.info(f"Task {args.task_name}: loading {split} from {cached_data_file}...")
            with open(cached_data_file, 'r') as f:
                idx_data = json.load(f)
        else:
            log.info(f"Task {args.task_name}: processing {split} from scratch...")

            tok_data = load_data_and_tokenize(args.data_dir, split, task_name, special_tokens)

            idx_data = index(tokenizer, tok_data)

            log.info(f"\tSaving indexed data to {cached_data_file}...")
            with open(cached_data_file, 'w') as f:
                json.dump(idx_data, f)

        no_input_lm = args.no_input_lm_train if split == "train" else args.no_input_lm_eval
        tsr_data = tensorfy(idx_data,
                            max_seq_len=model.config.n_positions,
                            task_name=task_name,
                            special2idx=special_tokens_ids,
                            no_input_lm=no_input_lm,
                            eoi_idx=special_tokens_ids['eos_tok'])

        batch_size = train_batch_size if split == "train" else eval_batch_size
        split2data[split] = create_loader(tsr_data, batch_size)
    train_dataloader = split2data["train"]
    eval_dataloader = split2data["dev"]

    # Prepare optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(model, train_dataloader, args)

    # Train loop
    tb_writer = SummaryWriter(args.out_dir)
    global_step, nb_tr_example_visits, best_eval_loss = 0, 0, float('inf')
    patience_left = args.patience
    start_time = time.time()
    for epoch_no in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss, tr_batch_loss, nb_tr_steps = 0, 0, 0
        tqdm_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(tqdm_bar):
            batch = tuple(t.to(device) for t in batch)
            if args.task_name in MC_TASK_NAMES:
                input_ids, mc_token_ids, lm_labels, mc_labels = batch
                outs = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                loss = args.lm_coef * outs[0] + outs[1]
                nb_tr_steps += 1
            else:
                input_ids, lm_labels = batch
                outs = model(input_ids, labels=lm_labels)
                loss = outs[0]
                nb_tr_steps += len(lm_labels[lm_labels != -1])

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_batch_loss += loss.item()
            nb_tr_example_visits += input_ids.size(0)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * scheduler.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], nb_tr_example_visits)
                tb_writer.add_scalar('loss', tr_batch_loss, nb_tr_example_visits)
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(tr_batch_loss, scheduler.get_lr()[0])
                tr_batch_loss = 0

        # Validation
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            if args.task_name in MC_TASK_NAMES:
                input_ids, mc_token_ids, lm_labels, mc_labels = batch
                with torch.no_grad(): # NOTE(Alex): this is probably broken
                    outs = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                    lm_loss, mc_loss = outs[0], outs[1]
                    eval_batch_loss = args.lm_coef * lm_loss + mc_loss
                    mc_logits = model(input_ids, mc_token_ids)[1]

                mc_logits = mc_logits.detach().cpu().numpy()
                mc_labels = mc_labels.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

                eval_loss += eval_batch_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
            else:
                input_ids, lm_labels = batch
                with torch.no_grad():
                    outs = model(input_ids, labels=lm_labels)
                    lm_loss = outs[0]

                eval_loss += lm_loss.mean().item()
                nb_eval_steps += 1 #len(lm_labels[lm_labels != -1])

            nb_eval_examples += input_ids.size(0)

        eval_loss /= nb_eval_steps
        tb_writer.add_scalar('eval_loss', eval_loss, nb_tr_example_visits)
        result = {'eval_loss': eval_loss,
                  'train_loss': tr_loss / (nb_tr_steps / float(args.gradient_accumulation_steps))}
        if args.task_name in MC_TASK_NAMES:
            result['eval_accuracy'] = eval_accuracy / nb_eval_examples

        output_eval_file = os.path.join(args.out_dir, "eval_results_{}.txt".format(epoch_no))
        with open(output_eval_file, "w") as writer:
            log.info("***** Eval results *****")
            for key in sorted(result.keys()):
                log.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # Model saving and early stopping
        log.info(f'Epoch {epoch_no + 1} complete!')
        if eval_loss < best_eval_loss:
            print('Best loss so far! {} -> {}'.format(best_eval_loss, eval_loss))
            best_eval_loss = eval_loss
            save_model(model, tokenizer, args, args.out_dir, 'model_epoch_{}.bin'.format(epoch_no), True)
            patience_left = args.patience
        else:
            print('Loss up from best epoch: {} -> {}'.format(best_eval_loss, eval_loss))
            save_model(model, tokenizer, args, args.out_dir, 'model_epoch_{}.bin'.format(epoch_no), False)
            patience_left -= 1
            if patience_left <= 0:
                print('Ran out of patience. Stopping training.')
                break

    print('Completed training in {}s!'.format(time.time() - start_time))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
