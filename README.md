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

```python qg_utils.py --command extract_ans --data_dir ${DATA_DIR}```

#### Generating questions

To actually generate the questions, we rely on BART finetuned on NewsQA, implemented in fairseq.

To generate from these models, we must first preprocess the data (tokenize and binarize) using the following command:
```./scripts/aw/preprocess.sh preprocess```
Make sure to change `dat_dir` to point to the directory containing your files, which should be named `{train, valid, test}.txt`

TODO(Alex): make sure we have the model dictionary file

Then to generate, use the following command.
```./scripts/aw/gen_sum.sh```

#### Filtering questions

Finally, we filter questions by quality using a number of heuristics.

Most importantly, we filter questions by enforcing answer consistency: 
We use a QA model to answer the generated questions, and if the predicted answer doesn't match the original answer, we throw out the question.
To do this, we need to run the QA model on the generated questions (see next section for commands to do so), which will produce an answer file.

To do the actual filtering, we run the following:
```python qg_utils.py --command filter_qsts --data_dir ${DATA_DIR}```


### Answering Questions

Model based on pytorch-pretrained-BERT (now `transformers`)

```./scripts/pt_qa.sh predict_extractive```


### Comparing Answers

Finally, to get the actual QAGS scores, we compare answers.

```python qa_utils.py --command compute_qags --src-ans-file ${src_ans_file} --trg-ans-file ${trg_ans_file}```



## Data

The crowdsourced annotations of summary sentences we collected are available in `data/mturk_{cnndm,xsum}.jsonl`.
Each line is an article, model-generated summary divided into sentences, and three annotations per sentence.
Each annotation is a binary choice of whether or not the summary sentence is factually supported by the article, 
as well as an anonymized annotator ID.

```data_stuff.py:compute_correlations_with_human```
