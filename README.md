# qags
Question Answering and Generation for Summarization

This is the code for the paper [Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://arxiv.org/abs/2004.04228), which appeared at ACL 2020.

The project spans multiple codebases [...]



## Usage

To compute QAGS scores, we need to

1. generate questions
2. answer questions
3. compare answers


### Generating Questions


#### Extracting answer candidates

We use an answer-conditional question generation model, so we first need to extract answer candidates.

```python qg_utils.py --command extract_ans --data_dir ${DATA_DIR}```

#### Generating questions

To actually generate the questions, we rely on BART finetuned on NewsQA, implemented in fairseq.
Code, model, and instructions for doing so are available [here](https://github.com/W4ngatang/qags/fairseq).

To generate from these models, we must first preprocess the data (tokenize and binarize) using the following command:
```./scripts/aw/preprocess.sh preprocess```

Make sure to change `dat_dir` to point to the directory containing your files, which should be named `{train, valid, test}.txt`

Then to generate, use the following command.
```./scripts/aw/gen_sum.sh```

#### Filtering questions

Finally, we filter questions by quality using a number of heuristics.
Most importantly, we filter questions by enforcing answer consistency: 
We use a QA model to answer the generated questions, and if the predicted answer doesn't match the original answer, we throw out the question.
To do this, we need to run the QA model on the generated questions (see next section), which will produce an answer file.
After having done so, TODO(Alex): instructions for using answer filtering once computed.

To do the actual filtering, we run the following:
```python qg_utils.py --command filter_qsts --data_dir ${DATA_DIR}```


### Answering Questions

TODO(Alex): data processing and formatting

To evaluate our QA models, use the following command to evaluate the model on `pred_file` and write the predictions to `out_dir/out_file`
Our models are based on `pytorch-pretrained-BERT` (now `transformers`) and pretrained checkpoints are located [here](TODO).
Make sure `model_dir` points to the QA model directory.
To compute QAGS scores, evaluate the QA model using the both the article as context and the summary as context, so you will need to run this command twice.

```
python finetune_pt_squad.py \
              --bert_model bert-large-uncased \
              --load_model_from_dir ${model_dir} \
              --version_2_with_negative \
              --do_lower_case \
              --do_predict \
              --predict_file ${pred_file} \
              --output_dir ${out_dir} \
              --prediction_file ${out_file} \
              --overwrite_output_dir
```


### Comparing Answers

Finally, to get the actual QAGS scores, we compare answers.

```python qa_utils.py --command compute-qags --src-ans-file ${src_ans_file} --trg-ans-file ${trg_ans_file}```



## Data

The crowdsourced annotations of summary sentences we collected are available in `data/mturk_{cnndm,xsum}.jsonl`.
Each line is an article, model-generated summary divided into sentences, and three annotations per sentence.
Each annotation is a binary choice of whether or not the summary sentence is factually supported by the article, 
as well as an anonymized annotator ID.

For CNNDM, the summarization model is Bottom-Up Summarization ([Gehrmann et al., 2017](https://arxiv.org/abs/1808.10792)).
For XSUM, the summarization model is BART ([Lewis et al., 2020](https://arxiv.org/abs/1910.13461)) finetuned on the XSUM training data.

```data_stuff.py:compute_correlations_with_human```


## Citation

If you use this code or data, please cite us.

@article{Wang_2020,
   title={Asking and Answering Questions to Evaluate the Factual Consistency of Summaries},
   url={http://dx.doi.org/10.18653/v1/2020.acl-main.450},
   DOI={10.18653/v1/2020.acl-main.450},
   journal={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
   publisher={Association for Computational Linguistics},
   author={Wang, Alex and Cho, Kyunghyun and Lewis, Mike},
   year={2020}
}

