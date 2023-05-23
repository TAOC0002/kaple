# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# batch=8
# accu=4
# lr=5e-6
# GPU='0'
# CUDA_VISIBLE_DEVICES=$GPU python examples/simcse.py \
# --model_name_or_path simcse \
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
# --num_train_epochs 30 \
# --metrics auc \
# --comment temp \
# --overwrite_output_dir \
# --mode bi

batch=8
accu=4
lr=5e-6
GPU='1'
CUDA_VISIBLE_DEVICES=$GPU python examples/kaple.py \
--model_name_or_path roberta-large \
--data_dir data/patent-sim-compact \
--preprocess_type read_examples_origin \
--output_dir ./proc_data/roberta_patentsim_compact \
--max_seq_length 512 \
--eval_steps 20 \
--per_gpu_train_batch_size $batch \
--gradient_accumulation_steps $accu \
--warmup_steps 0 \
--per_gpu_eval_batch_size $batch \
--learning_rate $lr \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--freeze_bert="" \
--freeze_adapter="" \
--adapter_size 768 \
--adapter_list "0,11,22" \
--adapter_skip_layers 0 \
--meta_fac_adaptermodel="./proc_data/adapter_pretraining/ddi-128/best-checkpoint/pytorch_model.bin" \
--meta_lin_adaptermodel="" \
--fusion_mode='concat' \
--num_train_epochs 30 \
--metrics auc \
--comment ddi-test \
--overwrite_output_dir \
--mode bi \
--pooling cls \
--loss bce 


# --meta_bertmodel="./proc_data/roberta_patentmatch/patentmatch_batch-8_lr-5e-06_warmup-0_epoch-6.0_concat/pytorch_bertmodel_4400_0.5436605821410952.bin"