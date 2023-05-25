# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Pre-train fac-adapter

task=cdr
GPU='0'
CUDA_VISIBLE_DEVICES=$GPU python ./adapters/patent-ner.py  \
        --model_type roberta \
        --model_name_or_path roberta-large \
        --data_dir ./data/cdr  \
        --output_dir proc_data/adapter_pretraining \
        --restore '' \
        --do_train  \
        --do_eval   \
        --evaluate_during_training 'True' \
        --task_name=$task     \
        --per_gpu_train_batch_size=4   \
        --per_gpu_eval_batch_size=4   \
        --num_train_epochs 5 \
        --max_seq_length 64 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --warmup_steps=1200 \
        --save_steps 20000 \
        --eval_steps 30 \
        --adapter_size 768 \
        --adapter_list "0,11,22" \
        --adapter_skip_layers 0 \
        --adapter_transformer_layers 2 \
        --meta_adapter_model="" \
        --comment='cdr-64-5epochs'