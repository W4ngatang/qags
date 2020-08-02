# Generate 

date=$(date '+%m-%d-%Y')

function generate() {
    data_path=/private/home/wangalexc/data/cnndailymail/processed
    model_path=/checkpoint/wangalexc/fairseq/summarization_best/ckpts2/checkpoint_best.pt
    out_dir=/checkpoint/wangalexc/fairseq/${date}
    log_file=/checkpoint/wangalexc/fairseq/${date}/cnndm-summaries.out
    mkdir -p ${out_dir}

    python summerization_generate.py ${data_path} --path ${model_path} --task summerization  --remove-bpe --gen-subset test --batch-size 1 --min-len 60 --max-target-positions 8000 --max-source-positions 8000 2>&1 | tee ${log_file}
}

generate
