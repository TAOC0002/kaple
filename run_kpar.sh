GPU='2'
CUDA_VISIBLE_DEVICES=$GPU python kpar/retriever.py \
--save_dir kpar/march23 \
--function construct_db \
--patent_model_ckpt proc_data/roberta_patentsim_compact/testing-sim/pytorch_model_bi_best_1.bin \
--pretrained_model_ckpt proc_data/roberta_patentsim_compact/testing-sim/pytorch_bertmodel_bi_best_1.bin \
--meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
--meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
--corpus_index_file data/patent-sim-compact/corpus_pool.pkl \
--corpus_content_file data/patent-sim/corpus_abstract.npy \
--query_file data/patent-sim-compact/test.jsonl \
--freeze_bert="" \
--freeze_adapter="" \