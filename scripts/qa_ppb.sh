# 

date=$(date '+%m-%d-%Y')

function train_v1_1() {
    export bert_version="bert-base-uncased" 
	export SQUAD_DIR=~/data/squad/v1.1/original
	export OUT_DIR=/checkpoint/wangalexc/ppb/${bert_version}/squad_v1_1/${date}
    mkdir -p ${OUT_DIR}

	python run_squad.py \
	  --bert_model ${bert_version}  \
	  --do_train \
	  --do_predict \
	  --do_lower_case \
	  --train_file $SQUAD_DIR/train-v1.1.json \
	  --predict_file $SQUAD_DIR/dev-v1.1.json \
	  --train_batch_size 64 \
	  --learning_rate 3e-5 \
	  --num_train_epochs 10.0 \
	  --max_seq_length 384 \
	  --doc_stride 128 \
	  --output_dir ${OUT_DIR}
}

function train_v2_0() {
    export bert_version="bert-base-uncased" 
	export SQUAD_DIR=~/data/squad/v2.0/original
	export OUT_DIR=/checkpoint/wangalexc/ppb/${bert_version}/squad_v2_0/${date}
    mkdir -p ${OUT_DIR}

	python run_squad.py \
	  --bert_model bert-base-uncased \
	  --do_train \
	  --do_predict \
	  --do_lower_case \
	  --train_file $SQUAD_DIR/train.json \
	  --predict_file $SQUAD_DIR/dev.json \
      --version_2_with_negative \
	  --train_batch_size 12 \
	  --learning_rate 3e-5 \
	  --num_train_epochs 2.0 \
	  --max_seq_length 384 \
	  --doc_stride 128 \
	  --output_dir ${OUT_DIR}
}

train_v1_1
