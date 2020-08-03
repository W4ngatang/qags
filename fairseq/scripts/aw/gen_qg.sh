# Generate 

gpu_id=${1:-0}
sampling=0
diverse=0

function generate() {

    model_path=${CKPTS}/fairseq/qags/qg_best.pt
    data_path=${PROC}/cnndailymail/fseq/labeled-subset/tmp/processed
    date=$(date '+%m-%d-%Y')

    min_len=5
    max_len_const=1 # max len is computed as ax + b where x is src len
    max_len_scale=1 # max len is computed as ax + b where x is src len
    beam_width=5
    n_hyps=10
    topk=10
    out_dir=${CKPTS}/fairseq/${date}
    out_file=${out_dir}/qst${beam_width}.tmp
    mkdir -p ${out_dir}

    if [ ${sampling} -eq 1 ]; then
        out_file=${out_file}.sampling.txt
    else
        out_file=${out_file}.txt
    fi

    if [ ${sampling} -eq 1 ]; then
        python fairseq/summerization_generate.py ${data_path} --device-id ${gpu_id} --path ${model_path} --task summerization  --remove-bpe --gen-subset test --batch-size 1 --min-len ${min_len} --max-len-a ${max_len_scale} --max-len-b ${max_len_const} --max-target-positions 8000 --max-source-positions 8000 --beam ${beam_width} --sampling --nbest ${n_hyps} --sampling-topk ${topk} 2>&1 | tee ${out_file}
    elif [ ${diverse} -eq 1 ]; then
        python fairseq/summerization_generate.py ${data_path} --device-id ${gpu_id} --path ${model_path} --task summerization  --remove-bpe --gen-subset test --batch-size 1 --min-len ${min_len} --max-len-a ${max_len_scale} --max-len-b ${max_len_const} --max-target-positions 8000 --max-source-positions 8000 --beam ${beam_width} --diverse-beam-groups 5 --nbest ${n_hyps} 2>&1 | tee ${out_file}
    else
        python fairseq/summerization_generate.py ${data_path} --device-id ${gpu_id} --path ${model_path} --task summerization  --remove-bpe --gen-subset test --batch-size 1 --min-len ${min_len} --max-len-a ${max_len_scale} --max-len-b ${max_len_const} --max-target-positions 8000 --max-source-positions 8000 --beam ${beam_width} --nbest ${n_hyps} 2>&1 | tee ${out_file}

    fi
}

generate
