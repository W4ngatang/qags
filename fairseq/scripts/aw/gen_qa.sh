# Generate 

data=${1:-"summarization-sents"}
gpu_id=${2:-0}
sampling=${3:-0}

date=$(date '+%m-%d-%Y')
ckpt_prefix="/checkpoint/wangalexc/fairseq"
mkdir -p ${ckpt_prefix}/${date}

function generate() {
    data_path=/private/home/wangalexc/data/squad/v1.1/tokenized
    out_file=/checkpoint/wangalexc/fairseq/${date}/qa-squadv1.1/valid.json
    log_file=/checkpoint/wangalexc/fairseq/${date}/qa-squadv1.1/log.log
    model_path=/checkpoint/wangalexc/fairseq/qa-gpt2-best/checkpoint_best.pt

    python scripts/eval_squad.py ${data_path} --path ${model_path} --task squad --remove-bpe --gen-subset valid --log-format tqdm --out-file ${out_file}
    #python scripts/eval_squad.py ${data_path} --path ${model_path} --task squad --remove-bpe --gen-subset valid 2>&1 | tee ${out_file}
}

generate
