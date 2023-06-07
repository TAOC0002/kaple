GPU='0'
CUDA_VISIBLE_DEVICES=$GPU python kpar/new_retriever.py \
--save_dir kpar/demo \
--function query \
--topk 3 \
--patent_model_ckpt proc_data/roberta_patentsim_compact/kpar-base-kadapter/pytorch_model_best.bin \
--pretrained_model_ckpt proc_data/roberta_patentsim_compact/kpar-base-kadapter/pytorch_bertmodel_best.bin \
--meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
--meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
--corpus_index_file data/patent-sim-compact/new_corpus_pool.pkl \
--corpus_content_file data/patent-sim/corpus_abstract.npy \
--query_file data/patent-sim-compact/test.jsonl \
--freeze_bert="" \
--freeze_adapter="" \
--model_name_or_path roberta-large \
--overwrite_save_dir \

# GPU='0'
# CUDA_VISIBLE_DEVICES=$GPU python kpar/new_retriever.py \
# --save_dir kpar/simcse \
# --function query \
# --topk 5 \
# --pretrained_model_ckpt proc_data/roberta_patentsim_compact/simcse-baseline/pytorch_bertmodel_best.bin \
# --corpus_index_file data/patent-sim-compact/new_corpus_pool.pkl \
# --corpus_content_file data/patent-sim/corpus_abstract.npy \
# --query_file data/patent-sim-compact/test.jsonl \
# --freeze_bert="" \
# --freeze_adapter="" \
# --model_name_or_path simcse \
# --overwrite_save_dir \