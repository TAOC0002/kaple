# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# batch=8
# accu=4
# lr=5e-6
# GPU='1,2'
# CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node 2 ../examples/simcse-sts.py \
# --model_name_or_path simcse \
# --data_dir ../data/sts/sts12-16/all \
# --year '2016' \
# --score_range 5 \
# --min_steps 10 \
# --preprocess_type read_sts_examples \
# --output_dir  ../proc_data/roberta_sts \
# --max_seq_length 512 \
# --eval_steps 60 \
# --per_gpu_train_batch_size $batch \
# --gradient_accumulation_steps $accu \
# --meta_bertmodel="../proc_data/roberta_sts/temp/pytorch_bertmodel_best.bin" \
# --meta_classifier="../proc_data/roberta_sts/temp/classifier_best.bin" \
# --warmup_steps 0 \
# --per_gpu_eval_batch_size $batch \
# --learning_rate $lr \
# --adam_epsilon 1e-6 \
# --weight_decay 0 \
# --freeze_bert="" \
# --metrics spearmanr \
# --comment temp \
# --overwrite_output_dir \
# --mode cross \
# --pooling cls \
# --loss mse \
# --do_test

batch=8
accu=8
lr=5e-6
GPU='0,1'
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node 2 ../examples/sts.py \
--model_name_or_path roberta-large \
--data_dir ../data/sts/sts12-16/all \
--year '2016' \
--score_range 5 \
--min_steps 10 \
--preprocess_type read_sts_examples \
--output_dir ../proc_data/roberta_sts \
--max_seq_length 512 \
--eval_steps 60 \
--per_gpu_train_batch_size $batch \
--gradient_accumulation_steps $accu \
--per_gpu_eval_batch_size $batch \
--learning_rate $lr \
--freeze_bert="" \
--freeze_adapter="" \
--adapter_size 768 \
--adapter_list "0,11,22" \
--adapter_skip_layers 0 \
--meta_fac_adaptermodel="../proc_data/adapter_pretraining/june10-stsb-cross/best-checkpoint/pytorch_model.bin" \
--meta_et_adaptermodel="../pretrained_models/fac-adapter/pytorch_model.bin" \
--meta_lin_adaptermodel="../pretrained_models/lin-adapter/pytorch_model.bin" \
--optimize_et_loss="" \
--fusion_mode='concat' \
--metrics spearmanr \
--comment june10-our-orig-bi \
--overwrite_output_dir \
--mode bi \
--pooling cls \
--loss mse \
--do_train \
--do_eval \
--do_test \
--freeze_adapter=""

# --meta_bertmodel="../proc_data/roberta_sts/temp/pytorch_bertmodel_best.bin" \
# --meta_classifier="../proc_data/roberta_sts/temp/classifier_best.bin" \
# ../proc_data/adapter_pretraining/june10-stsb-cross/best-checkpoint/pytorch_model.bin
