# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT license.

# batch=8
# accu=2
# lr=5e-6
# GPU='0'
# CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_patentmatch_cross_encoder.py \
# --model_type roberta-large \
# --model_name_or_path roberta-large \
# --task_name patentsim \
# --do_train \
# --do_eval \
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
# --save_steps 2000 \
# --report_steps 20000000000 \
# --freeze_bert="" \
# --freeze_adapter="" \
# --adapter_size 768 \
# --adapter_list "0,11,22" \
# --adapter_skip_layers 0 \
# --meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
# --meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
# --fusion_mode='concat' \
# --num_train_epochs 4 \
# --metrics auc \
# --comment one_k_bi \
# --logging_steps 2000000000 \
# --mode cross \
# --overwrite_output_dir
# --meta_bertmodel="./proc_data/roberta_patentmatch/patentmatch_batch-8_lr-5e-06_warmup-0_epoch-6.0_concat/pytorch_bertmodel_4400_0.5436605821410952.bin"


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

batch=8
accu=4
lr=5e-6
GPU='0'
CUDA_VISIBLE_DEVICES=$GPU python examples/bi-encoder.py \
--model_name_or_path simcse \
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
--meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
--meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
--fusion_mode='concat' \
--num_train_epochs 8 \
--metrics auc \
--comment temp \
--overwrite_output_dir \
--no_cuda \
--mode bi
# --pooling cls \
# --loss bce \


# --meta_bertmodel="./proc_data/roberta_patentmatch/patentmatch_batch-8_lr-5e-06_warmup-0_epoch-6.0_concat/pytorch_bertmodel_4400_0.5436605821410952.bin"