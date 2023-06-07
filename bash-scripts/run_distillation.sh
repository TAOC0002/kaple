batch=8
lr=5e-6
GPU='0'
CUDA_VISIBLE_DEVICES=$GPU python examples/self-distillation.py \
--model_name_or_path roberta-large \
--data_dir data/patent-match/ultra-balanced \
--output_dir ./proc_data/roberta_patentsim_compact \
--output_folder mar24 \
--max_seq_length 512 \
--eval_steps 120 \
--per_gpu_train_batch_size $batch \
--gradient_accumulation_steps_cross 2 \
--gradient_accumulation_steps_bi 8 \
--warmup_steps 0 \
--per_gpu_eval_batch_size $batch \
--learning_rate $lr \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--freeze_adapter="" \
--adapter_size 768 \
--adapter_list "0,11,22" \
--adapter_skip_layers 0 \
--meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
--meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
--num_train_epochs 6 \
--metrics accuracy \
--overwrite_output_dir \
--cycles 2