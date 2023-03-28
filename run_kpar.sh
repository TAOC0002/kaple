GPU='0'
CUDA_VISIBLE_DEVICES=$GPU python kpar/retriever.py \
--save_dir kpar/march26 \
--function query \
--topk 5 \
--patent_model_ckpt proc_data/roberta_patentsim_compact/one_k_simcse/pytorch_model_best.bin \
--pretrained_model_ckpt proc_data/roberta_patentsim_compact/one_k_simcse/pytorch_bertmodel_best.bin \
--meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
--meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
--corpus_index_file data/patent-sim-compact/corpus_pool.pkl \
--corpus_content_file data/patent-sim/corpus_abstract.npy \
--query_file data/patent-sim-compact/test.jsonl \
--freeze_bert="" \
--freeze_adapter="" \
--model_name_or_path roberta-large \
--overwrite_save_dir \