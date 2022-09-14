# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

batch=2
accu=1
lr=1e-5
GPU='0'
CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_patentmatch_adapter.py \
--model_type roberta-large \
--model_name_or_path roberta-large \
--do_train \
--do_eval \
--data_dir data/patent-match/ultra-balanced \
--preprocess_type read_examples_origin \
--output_dir ./proc_data/roberta_patentmatch \
--max_seq_length 512 \
--eval_steps 200 \
--per_gpu_train_batch_size $batch \
--gradient_accumulation_steps $accu \
--warmup_steps 0 \
--per_gpu_eval_batch_size $batch \
--learning_rate $lr \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 20000 \
--save_steps 2000 \
--report_steps 20000000000 \
--freeze_bert="" \
--freeze_adapter="True" \
--adapter_size 768 \
--adapter_list "0,11,22" \
--adapter_skip_layers 0 \
--meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
--meta_lin_adaptermodel=""

