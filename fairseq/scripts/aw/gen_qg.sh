# Generate 

data=${1:-"trg"}
gpu_id=${2:-0}
sampling=${3:-0}
diverse=0

date=$(date '+%m-%d-%Y')
#ckpt_prefix="/checkpoint/wangalexc/fairseq"
ckpt_prefix="${CKPTS}/fairseq"
mkdir -p ${ckpt_prefix}/${date}

function generate() {
    split="test"
    dataset="cnndm"    

    #model_path=/checkpoint/wangalexc/fairseq/qg_sentcontext/checkpoint_best.pt

    qg_epoch="best"
    if [ ${qg_epoch} == "best" ]; then
        model_path=${CKPTS}/fairseq/qg_paracontext/checkpoint_best.pt
    else
        model_path=${CKPTS}/fairseq/qg_paracontext/checkpoint${qg_epoch}.pt
    fi
    model_path=${CKPTS}/fairseq/qags/qg_best.pt

    min_len=5
    max_len_const=1 # max len is computed as ax + b where x is src len
    beam_width=5
    n_hyps=10
    topk=10


    if [ "${data}" == "qg" ]; then
        data_path=/private/home/wangalexc/data/squad-qg/binarized_v2
        out_file=/checkpoint/wangalexc/fairseq/${date}/questions-squad
        max_len_scale=3 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "src" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/src2trg/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst.src-v2.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "trg" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/trg2src/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst.trg-v2.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "gen" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/gen2src/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst.gen.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "lstm" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/lstm2src-v2/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst.lstm-beam10.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "lstmsmall" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/lstmsmall2src-v2/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst.lstmsmall-beam10.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "lstmsmalltied" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/lstmsmalltied2src-v2/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst.lstmsmalltied-beam10.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "tfmr" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/tfmr2src-v2/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst.tfmr-beam10.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len

    elif [ "${data}" == "src-subset" ]; then
        #data_path=/private/home/wangalexc/data/cnndailymail/roberta/labeled-subset/src2trg/processed
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/labeled-subset/src2trg/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst${beam_width}-ckpt${qg_epoch}.src-subset.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "bus" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/labeled-subset/bus2src/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst${beam_width}-ckpt${qg_epoch}.bus-subset.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "fan" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/labeled-subset/fan2src/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst${beam_width}.fan-subset.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "pgc" ]; then
        data_path=/private/home/wangalexc/data/cnndailymail/fseq/labeled-subset/pgc2src/processed
        out_file=/checkpoint/wangalexc/fairseq/${date}/qst${beam_width}.pgc-subset.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len
    elif [ "${data}" == "tmp" ]; then
        data_path=${PROC}/cnndailymail/fseq/labeled-subset/src2trg/tmp/processed
        out_file=${CKPTS}/fairseq/${date}/qst${beam_width}.tmp.${dataset}.${split}
        max_len_scale=1 # max len is computed as ax + b where x is src len

    else
        echo "Test generation file not supported!"
        exit 0
    fi


    if [ ${sampling} -eq 1 ]; then
        out_file=${out_file}.sampling.txt
    else
        out_file=${out_file}.txt
    fi

    if [ ${sampling} -eq 1 ]; then
        python summerization_generate.py ${data_path} --device-id ${gpu_id} --path ${model_path} --task summerization  --remove-bpe --gen-subset test --batch-size 1 --min-len ${min_len} --max-len-a ${max_len_scale} --max-len-b ${max_len_const} --max-target-positions 8000 --max-source-positions 8000 --beam ${beam_width} --sampling --nbest ${n_hyps} --sampling-topk ${topk} 2>&1 | tee ${out_file}
        #python generate.py ${data_path} --path ${model_path} --task summarization --remove-bpe --gen-subset test --batch-size 1 --min-len ${min_len} --max-len-a ${max_len_scale} --max-len-b ${max_len_const} --max-target-positions 8000 --max-source-positions 8000 --beam ${beam_width} --sampling --nbest ${n_hyps} --sampling-topk ${topk} 2>&1 | tee ${out_file}
    elif [ ${diverse} -eq 1 ]; then
        python summerization_generate.py ${data_path} --device-id ${gpu_id} --path ${model_path} --task summerization  --remove-bpe --gen-subset test --batch-size 1 --min-len ${min_len} --max-len-a ${max_len_scale} --max-len-b ${max_len_const} --max-target-positions 8000 --max-source-positions 8000 --beam ${beam_width} --diverse-beam-groups 5 --nbest ${n_hyps} 2>&1 | tee ${out_file}
    else
        ## pretrained QG model
        python -m ipdb summerization_generate.py ${data_path} --device-id ${gpu_id} --path ${model_path} --task summerization  --remove-bpe --gen-subset test --batch-size 1 --min-len ${min_len} --max-len-a ${max_len_scale} --max-len-b ${max_len_const} --max-target-positions 8000 --max-source-positions 8000 --beam ${beam_width} --nbest ${n_hyps} 2>&1 | tee ${out_file}

        # For RoBERTa
        #python generate.py ${data_path} --path ${model_path} --task summarization --source-lang src --target-lang trg --bpe gpt2 --remove-bpe --gen-subset test --batch-size 1 --min-len ${min_len} --max-len-a ${max_len_scale} --max-len-b ${max_len_const} --max-target-positions 8000 --max-source-positions 8000 --beam ${beam_width} --diverse-beam-groups 5 --nbest ${n_hyps} 2>&1 | tee ${out_file}
    fi
}

generate
