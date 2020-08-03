# Preprocessed tokenized and possibly BPE-ed text

function preprocess() {
    #dict_file="/private/home/wangalexc/data/cnndailymail/fseq/dict.txt"
    #dict_file="${PROC}/cnndailymail/fseq/dict.txt"
    dict_file="models/fseq/dict.txt"
    data="tmp"

    #dat_dir="/private/home/wangalexc/data/cnndailymail/fseq/labeled-subset/${data}"
    dat_dir="${PROC}/cnndailymail/fseq/labeled-subset/${data}"
    tok_dir="${dat_dir}/tokenized"
    out_dir="${dat_dir}/processed"
    mkdir -p ${tok_dir}
    mkdir -p ${out_dir}

    # tokenize
    python fairseq/prepreprocess.py --bert-version bert-base-uncased --data-dir ${dat_dir} --out-dir ${tok_dir}

    # preprocess: index, binarize
    python fairseq/preprocess.py --source-lang src --target-lang trg \
                                 --testpref ${tok_dir}/test.tok \
                                 --destdir ${out_dir} --thresholdtgt 10 --thresholdsrc 10 \
                                 --srcdict ${dict_file} --tgtdict ${dict_file} \
                                 --padding-factor 1 --workers 48;
                                 #--trainpref ${tok_dir}/train.tok --validpref ${tok_dir}/valid.tok --testpref ${tok_dir}/test.tok \

}


function preprocess_roberta() {
    # raw text -> tokenized -> binarized
    #for mdl in bus fan pgc; do
    for mdl in src; do

        encoder_file="/private/home/wangalexc/data/cnndailymail/roberta/encoder.json"
        vocab_file="/private/home/wangalexc/data/cnndailymail/roberta/vocab.bpe"
        dict_file="/private/home/wangalexc/data/cnndailymail/roberta/dict.txt"

        data="src2trg"
        dat_dir="/private/home/wangalexc/data/cnndailymail/roberta/labeled-subset/${data}"

        echo "Indexing with GPT2 codes ..."
        for split in train valid test; do
            for inp in src trg; do
                python -m examples.roberta.multiprocessing_bpe_encoder \
                    --encoder-json ${encoder_file}  \
                    --vocab-bpe ${vocab_file} \
                    --inputs ${dat_dir}/${split}.${inp} \
                    --outputs ${dat_dir}/${split}.${inp}.bpe \
                    --workers 10 \
                    --keep-empty;
                done;
        done
        echo "    Done!"

        echo "Binarizing with fairseq ..."
        for inp in src trg; do
            #--source-lang ${inp} \
            out_dir=${dat_dir}/processed/${inp}
            mkdir -p ${out_dir} 

            python preprocess.py \
                --only-source \
                --trainpref ${dat_dir}/train.${inp}.bpe \
                --validpref ${dat_dir}/valid.${inp}.bpe \
                --testpref ${dat_dir}/test.${inp}.bpe \
                --destdir ${out_dir} \
                --workers 10 \
                --srcdict ${dict_file};
        done
        echo "    Done!"

    done

}


function prepreprocess() {
    dict_file="/private/home/wangalexc/data/cnndailymail/fseq/dict.txt"

    #data_dir="/private/home/wangalexc/data/cnndailymail/fseq/gen2src/"
    #out_dir="/private/home/wangalexc/data/cnndailymail/fseq/gen2src/tokenized"

    data_dir="/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/src2trg"
    out_dir="/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/src2trg/tokenized"
    mkdir -p ${out_dir}

    python prepreprocess.py --bert-version bert-base-uncased --data-dir ${data_dir} --out-dir ${out_dir}
}

function fseq_preprocess() {
    dict_file="/private/home/wangalexc/data/cnndailymail/fseq/dict.txt"

    #data_dir="/private/home/wangalexc/data/cnndailymail/fseq/gen2src/tokenized"
    #out_dir="/private/home/wangalexc/data/cnndailymail/fseq/gen2src/processed"

    data="tfmr2src"
    data_dir="/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/${data}/tokenized"
    out_dir="/private/home/wangalexc/data/cnndailymail/fseq/onmt-models/${data}/processed"

    python preprocess.py --source-lang src --target-lang trg \
                         --trainpref ${data_dir}/train.tok --validpref ${data_dir}/valid.tok --testpref ${data_dir}/test.tok \
                         --destdir ${out_dir} --thresholdtgt 10 --thresholdsrc 10 \
                         --srcdict ${dict_file} --tgtdict ${dict_file} \
                         --padding-factor 1 --workers 48

    # Yinhan command
    #python preprocess.py --only-source 
    #                     --trainpref ~yinhanliu/data/bookwiki_2s/bert_data_split_doc2/train.txt --validpref ~yinhanliu/data/bookwiki_2s/bert_data_split_doc2/valid.txt 
    #                     --srcdict ~yinhanliu/data/bookwiki_2s/bert_data_split_doc2/dict.txt --destdir data/toy --workers 48 --padding-factor 1
}

function squad_preprocess() {
    #data_dir="/private/home/wangalexc/data/squad/v1.1/original"
    #out_dir="/private/home/wangalexc/data/squad/v1.1/tokenized"

    data_dir="/private/home/wangalexc/projects/qags/data/"
    out_dir="/private/home/wangalexc/projects/qags/data/"

    python scripts/data/preprocess_squad.py --output ${out_dir} --inputs ${data_dir}/train-v1.1.json ${data_dir}/dev-v1.1.json 
}

if [ $1 == "prepreprocess" ]; then
    prepreprocess
elif [ $1 == "fseq-preprocess" ]; then
    fseq_preprocess
elif [ $1 == "squad-preprocess" ]; then
    squad_preprocess
elif [ $1 == "preprocess" ]; then
    preprocess
elif [ $1 == "preprocess-roberta" ]; then
    preprocess_roberta
else
    echo "Command not recognized!"
    exit 0
fi
