#!/bin/bash

#SBATCH --job-name=qg-lstmsmall
#SBATCH --output=/checkpoint/wangalexc/fairseq/07-18-2019/slurm.out
#SBATCH --error=/checkpoint/wangalexc/fairseq/07-18-2019/slurm.err
#SBATCH --mail-user=wangalexc@fb.com
#SBATCH --mail-type=end

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00

./scripts/aw/gen_qg.sh lstmsmall
