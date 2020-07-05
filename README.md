# qags
Question Answering and Generation for Summarization

This is the code for the paper [Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://arxiv.org/abs/2004.04228), which will appear at ACL 2020.

The project spans multiple codebases [...]

## Usage

To compute QAGS scores, we need to

1. generate questions
2. answer questions
3. compare answers

### Generating Questions

Model based on fairseq: currently in `ckpt_fairseq/fairseq_backup/qg_paracontext/checkpoint_best.pt`

#### Extracting answer candidates

We use an answer-conditional question generation model, so we first need to extract answer candidates.

```data_stuff.py:extract_ans
```

#### Generating questions

To actually generate the questions, we rely on BART finetuned on NewsQA, implemented in fairseq.

```./scripts/aw/gen_sum.sh```

#### Filtering questions

Finally, we filter questions by quality using a number of heuristics.

```data_stuff.py:filter_qsts```

### Answering Questions

Model based on pytorch-pretrained-BERT (now `transformers`)

`./scripts/pt_qa.sh predict_extractive`

### Comparing Answers

`data_stuff.py:get_qags_scores`

## Data

`data_stuff.py:compute_correlations_with_human`
