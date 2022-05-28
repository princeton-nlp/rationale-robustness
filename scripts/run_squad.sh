#!/bin/bash
model_option=$1

if [[ "$model_option" == "fc" ]]; then
    python -m rrtl.run_squad --run-name squad_fc \
                             --model-type fc_squad \
                             --dataset-name squad \
                             --batch_size 32 \
                             --num_epoch 3
elif [[ "$model_option" == "vib" ]]; then
    python -m rrtl.run_squad --run-name squad_vib \
                             --model-type vib_squad_sent \
                             --dataset-name squad \
                             --batch_size 32 \
                             --num_epoch 3 \
                             --dataparallel \
                             --encoder-type distilbert-base-uncased \
                             --decoder-type distilbert-base-uncased \
                             --pi 0.75 \
                             --beta 0.1

elif [[ "$model_option" == "spectra" ]]; then
    python -m rrtl.run_squad --run-name squad_spectra \
                             --model-type spectra \
                             --dataset-name squad \
                             --batch_size 32 \
                             --num_epoch 3 \
                             --dataparallel \
                             --encoder-type distilbert-base-uncased \
                             --decoder-type distilbert-base-uncased \
                             --tau 1.0 \
                             --temperature 0.1
else
    echo "Task should be one of the options: [fc | vib | spectra]"
fi
