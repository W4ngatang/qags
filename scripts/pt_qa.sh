# Do QA stuff with pytorch pretrained BERT

bert_version=${2:-"bert-large-uncased"}
qst_src=${3:-"src"}
txt_fld=${4:-"trg"}
gen_mdl=${5:-"fseq"}
date=$(date '+%m-%d-%Y')

function train_squad_v1_1() {
	export SQUAD_DIR=~/data/squad/v1.1/original
	export OUT_DIR=/checkpoint/wangalexc/ppb/${bert_version}/squad_v1_1/${date}
    mkdir -p ${OUT_DIR}

	python finetune_pt_squad.py \
	  --bert_model ${bert_version}  \
      --do_train \
	  --do_predict \
	  --do_lower_case \
	  --train_file $SQUAD_DIR/train-v1.1.json \
	  --predict_file $SQUAD_DIR/dev-v1.1.json \
	  --train_batch_size 24 \
	  --learning_rate 3e-5 \
	  --num_train_epochs 2.0 \
	  --max_seq_length 384 \
	  --doc_stride 128 \
	  --output_dir ${OUT_DIR} \
      --overwrite_output_dir
}

function train_squad_v2_0() {
	export SQUAD_DIR=~/data/squad/v2.0/original
	export OUT_DIR=/checkpoint/wangalexc/ppb/${bert_version}/squad_v2_0/${date}
    mkdir -p ${OUT_DIR}

	python finetune_pt_squad.py \
	  --bert_model ${bert_version} \
      --do_train \
	  --do_predict \
	  --do_lower_case \
	  --train_file $SQUAD_DIR/train.json \
	  --predict_file $SQUAD_DIR/dev.json \
      --version_2_with_negative \
	  --train_batch_size 24 \
	  --learning_rate 3e-5 \
	  --num_train_epochs 2.0 \
	  --max_seq_length 384 \
	  --doc_stride 128 \
	  --output_dir ${OUT_DIR} \
      --overwrite_output_dir
}

function train_nlg() {
    model="gpt2-medium"
	data_dir="/private/home/wangalexc/data/squad/v2.0/original"
	out_dir="/checkpoint/wangalexc/ppb/${model}/squadv2_0_freetext/${date}"
    mkdir -p ${out_dir}

    batch_size=1

	python -m ipdb finetune_pt_lm.py \
        --model_name ${model} \
        --task_name squad-freetext \
	    --data_dir ${data_dir} \
	    --out_dir ${out_dir} \
        --overwrite_output_dir \
        --no_input_lm_train \
        --train_batch_size ${batch_size} \
        --reload_data
}

function evaluate_v1_1() {
	export data_file="/private/home/wangalexc/data/squad/v1.1/original/dev-v1.1.json"

	# bert base uncased
 	#export pred_file="/checkpoint/wangalexc/ppb/bert-base-uncased/squad_v1_1/06-25-2019/predictions.json"

	# bert large uncased
	export pred_file="/checkpoint/wangalexc/ppb/bert-large-uncased/squad_v1_1/06-25-2019/predictions.json"

	python evaluate-squad-v1-1.py ${data_file} ${pred_file}
}

function evaluate_v2_0() {
	export data_file="/private/home/wangalexc/data/squad/v2.0/original/dev.json"

	# bert base uncased
 	#export pred_file="/checkpoint/wangalexc/ppb/bert-base-uncased/squad_v2_0/06-25-2019/predictions.json"

	# bert large uncased
	export pred_file="/checkpoint/wangalexc/ppb/bert-large-uncased/squad_v2_0/06-25-2019/predictions.json"

	python evaluate-squad-v2-0.py ${data_file} ${pred_file}
}

function predict_extractive() {

    date="06-25-2019"
    squad_version="v2_0"

    pred_file="/private/home/wangalexc/projects/qags/data/${gen_mdl}/qst-${qst_src}.cnndm-${txt_fld}.json"
	mdl_dir="/checkpoint/wangalexc/ppb/${bert_version}/squad_${squad_version}/${date}-${squad_version}/"
	out_dir="/checkpoint/wangalexc/ppb/${bert_version}/squad_${squad_version}/${date}-${squad_version}/${gen_mdl}"
	out_file="${out_dir}/prd.qst-${qst_src}.cnndm-${txt_fld}.json"
    mkdir -p ${out_dir}

    # NOTE(Alex): maybe need --version_2_with_negative \
	python finetune_pt_squad.py \
	  --bert_model ${bert_version} \
	  --do_predict \
	  --do_lower_case \
	  --predict_file ${pred_file} \
	  --max_seq_length 384 \
	  --doc_stride 128 \
	  --output_dir ${out_dir} \
      --prediction_file ${out_file} \
      --overwrite_output_dir \
      --load_model_from_dir ${mdl_dir} \
      --version_2_with_negative

}

function evaluate_answers() {
    squad_version="2_0"

    src_file=/checkpoint/wangalexc/ppb/${bert_version}/squad_v${squad_version}/06-25-2019-v${squad_version}/${gen_mdl}/prd.qst-${qst_src}.cnndm-src.json
    trg_file=/checkpoint/wangalexc/ppb/${bert_version}/squad_v${squad_version}/06-25-2019-v${squad_version}/${gen_mdl}/prd.qst-${qst_src}.cnndm-${txt_fld}.json

    echo "Gold file: ${src_file}"
    echo "Pred file: ${trg_file}"
    python eval_ppb_answers.py --source-ans-file ${src_file} --target-ans-file ${trg_file};
    #python -m ipdb evaluate-squad-v2-0.py --data-file ${src_file} --pred-file ${trg_file} --verbose

}

function evaluate_all_answers() {
    squad_version="2_0"
    gen_mdl="lstm"

    for qst_src in src gen trg; do 
        for txt_fld in trg gen; do
            for bert_version in bert-base-uncased bert-large-uncased bert-large-uncased-whole-word-masking; do
                src_file=/checkpoint/wangalexc/ppb/${bert_version}/squad_v${squad_version}/06-25-2019-v${squad_version}/${gen_mdl}/prd.qst-${qst_src}.cnndm-src.json
                trg_file=/checkpoint/wangalexc/ppb/${bert_version}/squad_v${squad_version}/06-25-2019-v${squad_version}/${gen_mdl}/prd.qst-${qst_src}.cnndm-${txt_fld}.json

                echo "Gold file: ${src_file}"
                echo "Pred file: ${trg_file}"
                python eval_ppb_answers.py --source-ans-file ${src_file} --target-ans-file ${trg_file};
                #python -m ipdb evaluate-squad-v2-0.py --data-file ${src_file} --pred-file ${trg_file} --verbose
            done;
        done;
    done
}

if [ $1 == "train-squadv1.1" ]; then
    train_squad_v1_1
elif [ $1 == "train-squadv2.0" ]; then
    train_squad_v2_0
elif [ $1 == "train-nlg" ]; then
    train_nlg
elif [ $1 == "evaluate-v1.1" ]; then
    evaluate_v1_1
elif [ $1 == "evaluate-v2.0" ]; then
    evaluate_v2_0
elif [ $1 == "predict-extractive" ]; then
    predict_extractive
elif [ $1 == "evaluate" ]; then
    evaluate_answers
elif [ $1 == "evaluate-all" ]; then
    evaluate_all_answers
else
    echo "Command not found"
fi
