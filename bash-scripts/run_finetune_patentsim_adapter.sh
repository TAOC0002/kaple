# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# batch=8
# accu=4
# lr=5e-6
# GPU='2'
# CUDA_VISIBLE_DEVICES=$GPU python ../examples/simcse.py \
# --model_name_or_path bert \
# --data_dir data/patent-sim-compact \
# --preprocess_type read_examples_origin \
# --output_dir ./proc_data/roberta_patentsim_compact \
# --max_seq_length 512 \
# --eval_steps 20 \
# --per_gpu_train_batch_size $batch \
# --gradient_accumulation_steps $accu \
# --warmup_steps 0 \
# --per_gpu_eval_batch_size $batch \
# --learning_rate $lr \
# --adam_epsilon 1e-6 \
# --weight_decay 0 \
# --freeze_bert="" \
# --num_train_epochs 15 \
# --metrics auc \
# --comment parallel \
# --overwrite_output_dir \
# --mode bi

batch=8
accu=4
lr=5e-6
GPU='2'
CUDA_VISIBLE_DEVICES=$GPU python ../examples/kaple.py \
--model_name_or_path roberta-large \
--data_dir ../data/patent-sim-compact \
--preprocess_type read_examples_origin \
--output_dir ../proc_data/roberta_patentsim_compact \
--max_seq_length 512 \
--eval_steps 20 \
--per_gpu_train_batch_size $batch \
--gradient_accumulation_steps $accu \
--warmup_steps 0 \
--per_gpu_eval_batch_size $batch \
--num_train_epochs 30 \
--learning_rate $lr \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--freeze_bert="" \
--freeze_adapter="" \
--adapter_size 768 \
--adapter_list "0,11,22" \
--adapter_skip_layers 0 \
--meta_fac_adaptermodel="../proc_data/adapter_pretraining/june14-nli/best-checkpoint/pytorch_model.bin" \
--meta_et_adaptermodel="../pretrained_models/fac-adapter/pytorch_model.bin" \
--meta_lin_adaptermodel="../pretrained_models/lin-adapter/pytorch_model.bin" \
--optimize_et_loss="" \
--fusion_mode='concat' \
--metrics auc \
--comment june14-nli-adapter \
--overwrite_output_dir \
--mode cross \
--pooling cls \
--loss bce \
--do_train \
--do_eval \
--do_test \
--freeze_adapter=""

# --meta_bertmodel="./proc_data/roberta_patentsim_compact/cdr-ddi-test/pytorch_bertmodel_best.bin" \
# --meta_patentmodel="./proc_data/roberta_patentsim_compact/cdr-ddi-test/pytorch_model_best.bin" \
# --meta_bertmodel="./proc_data/roberta_patentmatch/patentmatch_batch-8_lr-5e-06_warmup-0_epoch-6.0_concat/pytorch_bertmodel_4400_0.5436605821410952.bin"
# "./pretrained_models/lin-adapter/pytorch_model.bin"
# "./pretrained_models/fac-adapter/pytorch_model.bin"
