from __future__ import absolute_import, division, print_function

import logging
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import faiss
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from util import set_seed, PretrainedModel, AdapterModel, patentModel, load_pretrained_adapter, cosine_sim, load_model, sigmoid
from parser import parse
from embeddings import compute_embed_pool, compute_query_pool
from torch.nn.functional import cosine_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse(parser)

    try:
        os.makedirs(args.save_subdir)
    except:
        pass
    if os.path.exists(args.save_subdir) and os.listdir(args.save_subdir) and not args.overwrite_save_dir:
        raise ValueError("Directory ({}) already exists and is not empty. Use --overwrite_save_dir to overcome.".format(args.save_dir))

    # Set up cuda
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        if device == torch.device("cuda"):
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 0
        # print(torch.cuda.device_count(), args.n_gpu, device)
    else: # Distributed
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Set up logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    
    # Set seed
    set_seed(args)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    pretrained_model = PretrainedModel(args)
    if args.meta_fac_adaptermodel:
        fac_adapter = AdapterModel(args, pretrained_model.config)
        fac_adapter = load_pretrained_adapter(fac_adapter,args.meta_fac_adaptermodel)
    else:
        fac_adapter = None
    if args.meta_et_adaptermodel:
        et_adapter = AdapterModel(args, pretrained_model.config)
        et_adapter = load_pretrained_adapter(et_adapter,args.meta_et_adaptermodel)
    else:
        et_adapter = None
    if args.meta_lin_adaptermodel:
        lin_adapter = AdapterModel(args, pretrained_model.config)
        lin_adapter = load_pretrained_adapter(lin_adapter,args.meta_lin_adaptermodel)
    else:
        lin_adapter = None
    patent_model = patentModel(args, pretrained_model.config,fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter, max_seq_length=args.max_seq_length, pooling="cls", loss="bce")
    pretrained_model.to(args.device)
    patent_model.to(args.device)
    logger.info('Load pre-trained bert model state dict from {}'.format(args.pretrained_model_ckpt))
    pretrained_model = load_model(args.pretrained_model_ckpt, pretrained_model)
    logger.info('Load pertrained patent model state dict from {}'.format(args.patent_model_ckpt))
    patent_model = load_model(args.patent_model_ckpt, patent_model)
    pretrained_model.eval()
    patent_model.eval()

    if args.function == 'construct_db':
        db_examples, embedding = compute_embed_pool(args.corpus_index_file, args.corpus_content_file, tokenizer, args.save_dir, pretrained_model, 
                           patent_model, max_seq_length=args.max_seq_length)
        with open(os.join(args.save_dir, 'db_examples.pkl'), 'wb') as f:
            pickle.dump(db_examples, f)
        np.save(os.join(args.save_dir, 'db_embedding.npy'), embedding)
    
    elif args.function == 'query':
        db_embeddings = np.load(os.join(args.save_dir, 'db_embedding.npy'), encoding="latin1")
        with open(os.join(args.save_dir, 'db_examples.pkl'), 'rb') as f:
            db_examples = pickle.load(f)
        query_examples, embedding = compute_query_pool(args.query_file, tokenizer, args.save_dir, pretrained_model, patent_model, args.max_seq_length)
        N = embedding.shape[0]
        dim = embedding.shape[1]

        # faiss
        nlist = 32
        res = faiss.StandardGpuResources()  # use a single GPU
        quantizer = faiss.IndexFlatL2(dim)  # the other index
        index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)
        assert not gpu_index_ivf.is_trained
        gpu_index_ivf.train(db_embeddings) 
        assert gpu_index_ivf.is_trained

        gpu_index_ivf.add(db_embeddings)
        print(gpu_index_ivf.ntotal)
        k = 5                         
        D, I = gpu_index_ivf.search(embedding, k) 

        # refinement
        embedding = embedding.unsqueeze(dim=1).expand(N, k, dim)
        for i in range(N):
            for j in range(k):
                if j == 0:
                    res_j = db_embeddings[I[i,j]].unsqueeze(0)
                else:
                    res_j = torch.cat([res_j, db_embeddings[I[i,j]].unsqueeze(0)], axis=0)
            if i == 0:
                refined_embedding = res_j.unsqueeze(0)
            else:
                refined_embedding = torch.cat([refined_embedding, res_j.unsqueeze(0)], axis=0)
        cosines = cosine_similarity(embedding, refined_embedding, dim=2)
        sorted, indices = torch.sort(cosines)
        ranking_tensor = torch.cat(tuple([torch.index_select(I[i], 0, indices[i]).unsqueeze(0) for i in range(10)]), 0) # in terms of numbering
        ranking_list = [[db_examples[y.item()].index for y in x] for x in ranking_tensor] # in terms of index

        # commpute mrr@k, k = 5
        # Evaluate on all test data
        hits = [[i+1 for i, e in enumerate(ranking_list[j]) if e in query_examples[j].ground_truth] for j in range(len(ranking_list))] # [[1, 2], [3], [3], [2], [2, 3], [2, 3]]
        mrr_score = torch.tensor([1/min(hit) if len(hit) else 0 for hit in hits]).mean().item() # 0.52777

        # compute map
        # Evaluate on a subset of test data where the no. of ground truths >= 5
        cutoff = i = 0
        while cutoff < len(query_examples):
            if len(query_examples[i].ground_truth) > k:
                break
            cutoff += 1
        map_hits = hits[:cutoff]
        map_func = lambda y: torch.tensor([torch.tensor([(i+1)/x[i] for i in range(len(x))]).mean() for x in y]).mean().item() # 0.55555
        map_score = map_func(map_hits)
        logger.info('MRR@5: {}, MAP@5: {}'.format(mrr_score, map_score))
        
