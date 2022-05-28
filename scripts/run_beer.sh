#!/bin/bash
model_option=$1

if [[ "$model_option" == "fc" ]]; then
    python -m rrtl.run_eraser --run-name beer_fc \
                                 --dataset-name beer \
                                 --model-type fc_beer \
                                 --lr 1e-5 \
                                 --dataparallel \
                                 --batch_size 16 \
                                 --num_epoch 20 \
                                 --grad_accumulation_steps 4
elif [[ "$model_option" == "vib" ]]; then
    python -m rrtl.run_eraser --run-name beer_vib \
                                 --dataset-name beer \
                                 --model-type vib_beer_token \
                                 --lr 1e-5 \
                                 --dataparallel \
                                 --batch_size 16 \
                                 --num_epoch 20 \
                                 --grad_accumulation_steps 4 \
                                 --pi 0.1 \
                                 --beta 0.01 \
                                 --tau 0.1
elif [[ "$model_option" == "spectra" ]]; then
    python -m rrtl.run_eraser --run-name beer_spectra \
                                 --dataset-name beer \
                                 --model-type spectra \
                                 --max_length 300 \
                                 --lr 1e-5 \
                                 --dataparallel \
                                 --batch_size 16 \
                                 --num_epoch 20 \
                                 --grad_accumulation_steps 4 \
                                 --budget_ratio 0.1 \
                                 --temperature 0.01 \
                                 --solver_iter 100
else
    echo "Task should be one of the options: [fc | vib | spectra]"
fi
