# Fine tune a BERT-like model for summarization
port=54186
DATE=`date +"%Y%m%d"`
num_nodes=1
WORLD_SIZE=8

data_path=/private/home/wangalexc/data/cnndailymail/processed
model_path=/checkpoint/wangalexc/fairseq/summarization_best/ckpts/checkpoint_best.pt
bert_path=/checkpoint/wangalexc/fairseq/summarization_best/ckpts3/checkpoint_best.pt
out_dir=/checkpoint/wangalexc/fairseq/summaries

python -O train.py ${data_path} --max-update 200000 --optimizer bert_adam  --lr-scheduler polynomial_decay --total-num-update 200000 --warmup 0.1 --lr 0.000015 --min-lr 1e-09 --clip-norm 0.0 --criterion summerization_loss  --max-tokens 2048 --task summerization  --arch ft_summerization --warmup 0.1 --bert-path ${bert_path} --max-target-positions 8000 --max-source-positions 8000 --save-dir ${out_dir}
