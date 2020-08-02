# Fine tune a BERT-like model for question generation

function train() {

    bert_path=/checkpoint/wangalexc/fairseq/bert-pretraining/20190520/checkpoint_best.pt
    #data_path=/private/home/wangalexc/data/squad-qg/binarized_sents
    data_path=/private/home/wangalexc/data/squad-qg/binarized_paras
    out_dir=/checkpoint/wangalexc/fairseq/qg_paracontext
    log_file=${out_dir}/train.log

    optimizer=bert_adam
    criterion=summerization_loss  
    task=summerization  
    arch=ft_summerization 
    #optimizer=adam #bert_adam
    #criterion=cross_entropy #summerization_loss  
    #task=translation #summerization  
    #arch=fconv_wmt_en_ro #ft_summerization 

    world_size=8
    batch_size=24
    max_epoch=25

    python train.py ${data_path} --max-update 200001 --optimizer ${optimizer} --lr-scheduler polynomial_decay --total-num-update 200000 --lr 0.000015 --min-lr 1e-09 --clip-norm 0.0 --criterion ${criterion} --max-tokens 2048 --task ${task} --arch ${arch} --max-target-positions 8000 --max-source-positions 8000 --save-dir ${out_dir} --warmup 0.1 --bert-path ${bert_path} --distributed-world-size ${world_size} --batch-size ${batch_size} --max-epoch ${max_epoch} # 2>&1 | tee ${log_file}
}

train
