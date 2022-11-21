#!/bin/bash
model_option=$1

if [[ "$model_option" == "fc" ]]; then
    python -m rrtl.run_eraser --run-name hotel_fc \
                                 --dataset-name hotel \
                                 --model-type fc_hotel \
                                 --lr 1e-5 \
                                 --dataparallel \
                                 --batch_size 16 \
                                 --num_epoch 20 \
                                 --grad_accumulation_steps 4
elif [[ "$model_option" == "vib" ]]; then
    python -m rrtl.run_eraser --run-name beer_vib \
                                 --dataset-name hotel \
                                 --model-type vib_hotel_token \
                                 --lr 1e-5 \
                                 --eval-interval 500 \
                                 --dataparallel \
                                 --batch_size 16 \
                                 --num_epoch 20 \
                                 --grad_accumulation_steps 4 \
                                 --pi 0.1 \
                                 --beta 0.01 \
                                 --tau 0.1

elif [[ "$model_option" == "spectra" ]]; then
    python -m rrtl.run_eraser --run-name hotel_spectra \
                                 --dataset-name hotel \
                                 --model-type spectra \
                                 --lr 1e-5 \
                                 --eval-interval 500 \
                                 --dataparallel \
                                 --batch_size 16 \
                                 --num_epoch 20 \
                                 --grad_accumulation_steps 4 \
                                 --budget_ratio 0.1 \
                                 --temperature 0.01 \
                                 --solver_iter 100
else
    echo "Task should be one of the options: [vib | spectra]"
fi
