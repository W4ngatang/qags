# qags
Question Answering and Generation for Summarization

Some helper scripts for project "Question Answering and Generation for Summarization".
The meat of the code is actually in fairseq.
This repo includes:
- `parse_fseq.py`: Parse fairseq output "logs" for the actual generations, outputting a JSONL containing the generations. Usage: 
- `allennlp_qa.py`: use AllenNLP pretrained QA models to evaluate questions and summaries. Usage: `python allennlp_qa.py 2>&1 | tee data/bidaf.log`
- `eval_answers.py`: Evaluate answer spans outputted by `allennlp_qa.py`.

## Usage

### Generating Questions

Model based on fairseq

#### Extracting answer candidates

`data_stuff.py:extract_ans`

#### Generating questions

In `fairseq-py`, `./scripts/aw/gen_sum.sh`

#### Filtering questions

`data_stuff.py:filter_qsts`

### Answering Questions

Model based on pytorch-pretrained-BERT

`./scripts/pt_qa.sh predict_extractive`

### Comparing Answers

`data_stuff.py:get_qags_scores`

## Data

`data_stuff.py:compute_correlations_with_human`
