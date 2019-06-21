# qags
Question Answering and Generation for Summarization

Some helper scripts for project "Question Answering and Generation for Summarization".
The meat of the code is actually in fairseq.
This repo includes:
- `parse_fseq.py`: Parse fairseq output "logs" for the actual generations, outputting a JSONL containing the generations. Usage: 
- `allennlp_qa.py`: use AllenNLP pretrained QA models to evaluate questions and summaries. Usage: `python allennlp_qa.py 2>&1 | tee data/bidaf.log`
- `eval_answers.py`: Evaluate answer spans outputted by `allennlp_qa.py`. Usage: 
