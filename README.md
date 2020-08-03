# qags
Question Answering and Generation for Summarization

This is the code for the paper [Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://arxiv.org/abs/2004.04228), which appeared at ACL 2020.


## Usage

To compute QAGS scores, we need to

1. generate questions
2. answer questions
3. compare answers


### 1. Generating Questions


#### Extracting answer candidates

We use an answer-conditional question generation model, so we first need to extract answer candidates.
Use the following command, where `data_file` is a text file containining an example per line and
`out_dir` is the directory to write the processed files to.
The script will produce `test.txt`, `test_{n_ans_per_txt}.txt`, `test_w_{n_ans_per_txt}ans.txt` 
in `out_dir`, which respectively contain the examples, the extracted answers, and the answers and examples formatted to
feed into the QG model.

```python qg_utils.py --command extract_ans --data_file ${data_file} --out_dir ${out_dir}```


#### Generating questions

To generate the questions, we rely on [BART](https://arxiv.org/abs/1910.13461) finetuned on [NewsQA](https://arxiv.org/abs/1611.09830), implemented in [`fairseq`](https://github.com/pytorch/fairseq).
A frozen version of `fairseq` for doing so is available in [`qags/fairseq`](https://github.com/W4ngatang/qags/fairseq).
Our pretrained QG model is available [here](TODO).

To generate from these models, we must first preprocess the data (tokenize and binarize) using the command:
`./fairseq/scripts/aw/preprocess.sh preprocess`.
In the script, make sure to change `dat_dir` to point to the directory containing your files.
The script expects `dat_dir` to contain `test.src` and `test.trg`, where `test.src` are the files that will actually 
be fed into the QG model to generate from; `test.trg` can be a dummy file with the same number of lines (e.g., a copy of `test.src`).

Then to generate, use command `./scripts/aw/gen_sum.sh`. 
Change `model_path` to point to the pretrained QG checkpoint,
`data_path` to the directory containing the processed data (typically the `processed` directory created during preprocessing),
and `out_file` for the file to log to.
Due to a code quirk, in `fairseq/fairseq/models/summerization_encoder_only.py`, set `HACK_PATH` (line 107) to the `best_pretrained_bert.pt` checkpoint.

Finally, extract the generated questions using `python qg_utils.py --command extract-qst` (TODO(Alex)).


### 2. Answering Questions

To prepare the QA data, use `python qa_utils.py --command format-data`. (TODO(Alex))

As part of this step, we filter questions by quality using a number of heuristics.
Most importantly, we filter questions by enforcing answer consistency: 
We use a QA model to answer the generated questions, and if the predicted answer doesn't match the original answer, we throw out the question.
To do this, we need to run the QA model on the generated questions, which will produce an answer file.
For this step, use the flag `--use_all_qsts` and then run the QA model on the resulting data file.

Once you have answers for each question, we need to compare the expected and predicted answers, 
which we do so by TODO(Alex): instructions for using answer filtering once computed.

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



### 3. Comparing Answers

Finally, to get the actual QAGS scores, we compare answers.
The following command will write the scores to `out_dir/qags_scores.txt`.

```
python qa_utils.py --command compute-qags \
                   --src-ans-file ${src_ans_file} \
                   --trg-ans-file ${trg_ans_file} \
                   --out-dir ${out_dir}
```



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

```
@article{wang2020asking,
   title={Asking and Answering Questions to Evaluate the Factual Consistency of Summaries},
   url={http://dx.doi.org/10.18653/v1/2020.acl-main.450},
   DOI={10.18653/v1/2020.acl-main.450},
   journal={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
   publisher={Association for Computational Linguistics},
   author={Wang, Alex and Cho, Kyunghyun and Lewis, Mike},
   year={2020}
}
```
