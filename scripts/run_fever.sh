#!/bin/bash
model_option=$1
base_path="/path/to/your/repo"

if [[ "$model_option" == "fc" ]]; then
    REPO_PATH=$base_path/rr/base/explainable_qa
    RUN_PATH=$base_path/rr/base/explainable_qa/baselines
    
    DATASET_NAME=fever
    BOTTLENECK_TYPE=full
    MODEL_NAME=fever_fc
    PRED_DIR=fever
    
    cd $RUN_PATH
    
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m torch.distributed.launch evidence_deploy.py \
        --data_dir $REPO_PATH/data/$DATASET_NAME \
        --output_dir $base_path/experiments/fever_fc \
        --model_type bert \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --classes REFUTES SUPPORTS \
        --eval_split val \
        --max_seq_length 512 \
        --max_query_length 32 \
        --local_rank -1 \
        --num_train_epochs 1 \
        --wait_step 10 \
        --evaluate_during_training \
        --logging_steps 500 \
        --save_steps 10000 \
        --dataset-name $DATASET_NAME \
        --bottleneck-type $BOTTLENECK_TYPE \
        --model-name $MODEL_NAME \
        --pred-dir $PRED_DIR
    cd $base_path

elif [[ "$model_option" == "vib" ]]; then
    REPO_PATH=$base_path/rr/base/explainable_qa
    RUN_PATH=$base_path/rr/base/explainable_qa/information_bottleneck
    
    DATASET_NAME=fever
    BOTTLENECK_TYPE=vib
    MODEL_NAME=fever_vib
    PRED_DIR=fever
    
    cd $RUN_PATH
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m torch.distributed.launch ib_train_sentence.py \
        --data_dir $REPO_PATH/data/$DATASET_NAME \
        --output_dir $base_path/experiments/$PRED_DIR/$BOTTLENECK_TYPE/$MODEL_NAME \
        --model_type distilbert_gated_sent \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --eval_split val \
        --max_seq_length 512 \
        --max_query_length 32 \
        --num_train_epochs 1 \
        --max_num_sentences 10 \
        --wait_step 10 \
        --evaluate_during_training \
        --logging_steps 500 \
        --save_steps 5000 \
        --local_rank -1 \
        --classes REFUTES SUPPORTS \
        --chunk_size 5 \
        --beta 1.0 \
        --threshold 0.4 \
        --dataset-name $DATASET_NAME \
        --bottleneck-type $BOTTLENECK_TYPE \
        --model-name $MODEL_NAME \
        --pred-dir $PRED_DIR
    cd $base_path

elif [[ "$model_option" == "spectra" ]]; then
    python -m rrtl.run_eraser --run-name fever_spectra \
                                 --dataset-name fever \
                                 --model-type spectra_fever_sent \
                                 --max_length 400 \
                                 --lr 5e-5 \
                                 --dataparallel \
                                 --batch_size 8 \
                                 --num_epoch 10 \
                                 --grad_accumulation_steps 4 \
                                 --budget_ratio 0.4 \
                                 --temperature 0.01 \
                                 --solver_iter 100
else
    echo "Task should be one of the options: [fc | vib | spectra]"
fi
