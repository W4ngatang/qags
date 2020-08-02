FAIRSEQ=/private/home/yinhanliu/fairseq-py-huggingface
port=54186
DATE=`date +"%Y%m%d"`
SWEEP_NAME=summerization
num_nodes=1
WORLD_SIZE=8
JOBSCRIPTS=scripts
mkdir -p ${JOBSCRIPTS}
#pretrain_path=/checkpoint/edunov/20190430/block.transformer.doc.512.eps-08.polynomial.bert_init.auto.600k/layers12/128/bi_cloze_0.0003/checkpoint_best.pt
#pretrain_path=/checkpoint/yinhanliu/20190520/block.transformer.auto.doc.512.fixed_order/layers12/32_0.001/checkpoint_best.pt
pretrain_path=/checkpoint/yinhanliu/20190520/block.transformer.auto.doc.512/layers12/32/l2r.lr_0.001.geometricP_0.1/checkpoint_best.pt
#pretrain_path=/checkpoint/yinhanliu/20190513/RISKYblock.transformer.auto.doc.512/layers12/32/l2r_0.001/checkpoint_best.pt
order=fix_order
queue=learnfair
JNAME=${SWEEP_NAME}
SAVE_ROOT=/checkpoint/yinhanliu/${DATE}/${SWEEP_NAME}/512.${ordering}
mkdir -p stdout stderr
for learn_rate in 0.00001
do
        SAVE=${SAVE_ROOT}_${learn_rate}
        mkdir -p ${SAVE}
        JNAME=${SWEEP_NAME}._learn_rate_${learn_rate}
        SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
        SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm
        echo "#!/bin/sh" > ${SCRIPT}
        echo "#!/bin/sh" > ${SLURM}
        echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
        echo "#SBATCH --output=stdout/${JNAME}.%j" >> ${SLURM}
        echo "#SBATCH --error=stderr/${JNAME}.%j" >> ${SLURM}
        echo "#SBATCH --signal=USR1@120" >> ${SLURM}
        echo "#SBATCH --partition=${queue}" >> ${SLURM}
        echo "#SBATCH --comment=emnlparchivedeadline" >> ${SLURM}
        echo "#SBATCH --nodes=${num_nodes} -C volta32gb" >> ${SLURM}
        #echo "#SBATCH --nodes=${num_nodes}" >> ${SLURM}
        echo "#SBATCH --ntasks-per-node=8" >> ${SLURM}
        echo "#SBATCH --mem=500000" >> ${SLURM}
        echo "#SBATCH --gres=gpu:8" >> ${SLURM}
        echo "#SBATCH --cpus-per-task 8" >> ${SLURM}
        echo "#SBATCH --time=4320" >> ${SLURM}
	echo "#SBATCH --constraint=volta32gb" >> ${SLURM}
        echo "srun sh ${SCRIPT}" >> ${SLURM}
        echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
        echo "{ " >> ${SCRIPT}
        echo "echo $SWEEP_NAME " >> ${SCRIPT}
        echo "cd $FAIRSEQ" >> ${SCRIPT}
	echo "python -O train.py /private/home/yinhanliu/data/cnndailymail/processed  --max-update 200000 --optimizer bert_adam --fp16  --skip-invalid-size-inputs-valid-test --lr-scheduler polynomial_decay --total-num-update 200000 --warmup 0.1 --lr ${learn_rate} --min-lr 1e-09 --clip-norm 0.0 --criterion summerization_loss  --max-tokens 3072 --task summerization  --arch ft_summerization --warmup 0.1 --bert-path ${pretrain_path}  --max-target-positions 8000 --max-source-positions 8000 --save-dir ${SAVE} --distributed-world-size ${WORLD_SIZE} --distributed-port ${port} " >> ${SCRIPT}
        echo "kill -9 \$\$" >> ${SCRIPT}
        echo "} & " >> ${SCRIPT}
        echo "child_pid=\$!" >> ${SCRIPT}
	echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
        echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
        echo "while true; do     sleep 1; done" >> ${SCRIPT}
        sbatch ${SLURM}
done
