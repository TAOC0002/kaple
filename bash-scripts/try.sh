batch=8
accu=8
lr=5e-6
GPU='0'
CUDA_VISIBLE_DEVICES=$GPU python ../examples/adaptformer/main.py \
--model_name_or_path roberta-large \
--data_dir ../data/sts/sts12-16/all \
--year '2016' \
--output_dir ../proc_data/roberta_sts \
--max_seq_length 512 \
--eval_steps 60 \
--num_train_epochs 15 \
--per_gpu_train_batch_size $batch \
--gradient_accumulation_steps $accu \
--per_gpu_eval_batch_size $batch \
--learning_rate $lr \
--comment july5 \
--overwrite_output_dir \
--do_train \
--do_eval