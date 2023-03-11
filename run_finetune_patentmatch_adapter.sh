# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# batch=8
# accu=4
# lr=5e-6
# GPU='1'
# CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_simcse_bi_encoder.py \
# --model_type roberta-large \
# --model_name_or_path roberta-large \
# --do_train \
# --do_eval \
# --data_dir data/patent-match/ultra-balanced \
# --preprocess_type read_examples_origin \
# --output_dir ./proc_data/roberta_patentmatch \
# --max_seq_length 512 \
# --eval_steps 200 \
# --per_gpu_train_batch_size $batch \
# --gradient_accumulation_steps $accu \
# --warmup_steps 0 \
# --per_gpu_eval_batch_size $batch \
# --learning_rate $lr \
# --adam_epsilon 1e-6 \
# --save_steps 2000 \
# --report_steps 20000000000 \
# --freeze_bert="" \
# --freeze_adapter="True" \
# --adapter_size 768 \
# --adapter_list "0,11,22" \
# --adapter_skip_layers 0 \
# --weight_decay 0 \
# --meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
# --meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
# --fusion_mode='concat' \
# --num_train_epochs 4 \
# --mode "bi" \
# --comment="concat_no_wd_roberta_segment_bi"

batch=8
accu=4
lr=5e-5
GPU='3'
CUDA_VISIBLE_DEVICES=$GPU python examples/bi-encoder.py \
--model_name_or_path roberta-large \
--data_dir data/patent-match/ultra-balanced \
--preprocess_type read_examples_origin \
--output_dir ./proc_data/roberta_patentmatch \
--max_seq_length 512 \
--eval_steps 120 \
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
--num_train_epochs 10 \
--metrics accuracy \
--comment cls_pooling_mse_10_epochs \
--mode bi \
--pooling cls \
--loss mse \