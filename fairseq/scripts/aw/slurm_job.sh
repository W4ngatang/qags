#!/bin/bash

#SBATCH --job-name=ppb_predict
#SBATCH --output=/checkpoint/wangalexc/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/slurm.out
#SBATCH --error=/checkpoint/wangalexc/ppb/bert-large-uncased-whole-word-masking/squad_v2_0/06-25-2019-v2_0/slurm.err
#SBATCH --mail-user=wangalexc@fb.com
#SBATCH --mail-type=end

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

# NOTE(Alex): make sure to change the sbatch out and log files above
bert_version="bert-large-uncased-whole-word-masking"
qst_src="src"
txt_fld="src"

#./scripts/qa_ppb.sh train-v1.1 ${bert_version}
#./scripts/qa_ppb.sh train-v2.0 ${bert_verison}
./scripts/qa_ppb.sh predict ${bert_version} ${qst_src} ${txt_fld}
