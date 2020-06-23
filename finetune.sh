export TRAIN_FILE=pofo.corpus

python ../../transformers/examples/language-modeling/run_language_modeling.py \
    --output_dir=pofo \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --mlm
