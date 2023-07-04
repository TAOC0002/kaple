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
# --num_train_epochs 15 \
# --min_steps 10 \
# --preprocess_type read_sts_examples \
# --output_dir  ../proc_data/roberta_sts \
# --max_seq_length 512 \
# --eval_steps 60 \
# --per_gpu_train_batch_size $batch \
# --gradient_accumulation_steps $accu \
# --warmup_steps 0 \
# --per_gpu_eval_batch_size $batch \
# --learning_rate $lr \
# --adam_epsilon 1e-6 \
# --weight_decay 0 \
# --freeze_bert="" \
# --metrics spearmanr \
# --comment june29-2016-new \
# --overwrite_output_dir \
# --mode cross \
# --pooling cls \
# --loss mse \
# --do_train \
# --do_eval

# batch=8
# accu=8
# lr=5e-6
# GPU='3'
# CUDA_VISIBLE_DEVICES=$GPU python ../examples/sts.py \
# --model_name_or_path roberta-large \
# --data_dir ../data/sts/sts12-16/all \
# --year '2016' \
# --score_range 5 \
# --min_steps 10 \
# --preprocess_type read_sts_examples \
# --output_dir ../proc_data/roberta_sts \
# --max_seq_length 512 \
# --eval_steps 60 \
# --num_train_epochs 15 \
# --per_gpu_train_batch_size $batch \
# --gradient_accumulation_steps $accu \
# --per_gpu_eval_batch_size $batch \
# --learning_rate $lr \
# --freeze_bert="" \
# --freeze_adapter="" \
# --adapter_size 768 \
# --adapter_list "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22" \
# --adapter_skip_layers 0 \
# --meta_fac_adaptermodel="../proc_data/adapter_pretraining/june10-stsb-cross/best-checkpoint/pytorch_model.bin" \
# --meta_et_adaptermodel="../pretrained_models/fac-adapter/pytorch_model.bin" \
# --meta_lin_adaptermodel="../pretrained_models/lin-adapter/pytorch_model.bin" \
# --optimize_et_loss="" \
# --fusion_mode='concat' \
# --metrics spearmanr \
# --comment june29-2016-every \
# --overwrite_output_dir \
# --mode cross \
# --pooling cls \
# --loss mse \
# --do_train \
# --do_eval \
# --freeze_adapter="" \
# --freeze_bert="True"


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
--num_train_epochs 15 \
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
--comment june29-2016-early \
--overwrite_output_dir \
--mode cross \
--pooling cls \
--loss mse \
--do_train \
--do_eval \
--freeze_adapter="" \
--freeze_bert="True"


# --meta_bertmodel="../proc_data/roberta_sts/temp/pytorch_bertmodel_best.bin" \
# --meta_classifier="../proc_data/roberta_sts/temp/classifier_best.bin" \

# --meta_bertmodel="../proc_data/roberta_sts/june16-2014-cross/pytorch_bertmodel_best.bin" \
# --meta_patentmodel="../proc_data/roberta_sts/june16-2014-cross/pytorch_bertmodel_best.bin" \