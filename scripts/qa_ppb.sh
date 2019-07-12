# Do QA stuff with pytorch pretrained BERT

bert_version=${2:-"bert-large-uncased"}
qst_src=${3:-"src"}
txt_fld=${4:-"trg"}
date=$(date '+%m-%d-%Y')

function train_v1_1() {
	export SQUAD_DIR=~/data/squad/v1.1/original
	export OUT_DIR=/checkpoint/wangalexc/ppb/${bert_version}/squad_v1_1/${date}
    mkdir -p ${OUT_DIR}

	python run_ppb_squad.py \
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

function train_v2_0() {
	export SQUAD_DIR=~/data/squad/v2.0/original
	export OUT_DIR=/checkpoint/wangalexc/ppb/${bert_version}/squad_v2_0/${date}
    mkdir -p ${OUT_DIR}

	python run_ppb_squad.py \
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

function predict() {
    #date="07-02-2019"
    #bert_version="bert-large-uncased-whole-word-masking-finetuned-squad"

    date="06-25-2019"
    bert_version="bert-large-uncased"

    squad_version="v2_0"

    pred_file=/private/home/wangalexc/projects/qags/data/qst-${qst_src}.cnndm-${txt_fld}.json
	out_dir=/checkpoint/wangalexc/ppb/${bert_version}/squad_${squad_version}/${date}-${squad_version}
    mkdir -p ${out_dir}

    # NOTE(Alex): maybe need --version_2_with_negative \
	python run_ppb_squad.py \
	  --bert_model ${bert_version} \
	  --do_predict \
	  --do_lower_case \
	  --predict_file ${pred_file} \
	  --max_seq_length 384 \
	  --doc_stride 128 \
	  --output_dir ${out_dir} \
      --overwrite_output_dir \
      --load_model_from_dir ${out_dir} \
      --version_2_with_negative

    mv ${out_dir}/predictions.json ${out_dir}/prd.qst-${qst_src}.cnndm-${txt_fld}.json
}

function evaluate_answers() {
    bert_version="bert-large-uncased"
    squad_version="1-1"
    question_source="sources"
    context="targets"

    #src_file=/checkpoint/wangalexc/ppb/bert-base-uncased/squad_v2_0/06-25-2019-v2/predictions.cnndm-sources.sources.json
    #trg_file=/checkpoint/wangalexc/ppb/bert-base-uncased/squad_v2_0/06-25-2019-v2/predictions.cnndm-sources.targets.json

    src_file=/checkpoint/wangalexc/ppb/${bert_version}/squad_v2_0/06-25-2019-v${squad_version}/predictions.cnndm-${question_source}.sources.json
    trg_file=/checkpoint/wangalexc/ppb/${bert_version}/squad_v2_0/06-25-2019-v${squad_version}/predictions.cnndm-${question_source}.targets.json

    python eval_ppb_answers.py --source-ans-file ${src_file} --target-ans-file ${trg_file}
}

if [ $1 == "train-v1.1" ]; then
    train_v1_1
elif [ $1 == "train-v2.0" ]; then
    train_v2_0
elif [ $1 == "evaluate-v1.1" ]; then
    evaluate_v1_1
elif [ $1 == "evaluate-v2.0" ]; then
    evaluate_v2_0
elif [ $1 == "predict" ]; then
    predict
elif [ $1 == "evaluate" ]; then
    evaluate_answers
else
    echo "Command not found"
fi
