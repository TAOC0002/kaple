from __future__ import absolute_import, division, print_function

import logging
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import faiss
import json
import sys
import time
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from examples.util import set_seed, PretrainedModel, AdapterModel, patentModel, load_pretrained_adapter, cosine_sim, load_model, sigmoid
from parser import parse
from embeddings import compute_embed_pool, compute_query_pool
from torch.nn.functional import cosine_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse(parser)

    try:
        os.makedirs(args.save_dir)
    except:
        pass
    if os.path.exists(args.save_dir) and os.listdir(args.save_dir) and not args.overwrite_save_dir:
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

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', add_special_tokens=True)
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
                           patent_model, logger, device, max_seq_length=args.max_seq_length)
        with open(os.path.join(args.save_dir, 'db_examples.pkl'), 'wb') as f:
            pickle.dump(db_examples, f)
        np.save(os.path.join(args.save_dir, 'db_embedding.npy'), embedding)
        logger.info('Finished!')
    
    elif args.function == 'query':
        st = time.time()
        db_embeddings = np.load(os.path.join(args.save_dir, 'db_embedding.npy'), encoding="latin1")
        with open(os.path.join(args.save_dir, 'db_examples.pkl'), 'rb') as f:
            db_examples = pickle.load(f)
        query_examples, embedding = compute_query_pool(args.query_file, tokenizer, args.save_dir, pretrained_model, patent_model, logger,
                                                       device, max_seq_length=args.max_seq_length)
        N = embedding.shape[0]
        dim = embedding.shape[1]
        logger.info('Time elapsed in loading db and constructing query embeddings: {:.2f} seconds'.format(time.time()-st))

        ## Use an IVF Index
        # faiss
        # st = time.time()
        # nlist = args.nlist
        # res = faiss.StandardGpuResources()  # use a single GPU
        # quantizer = faiss.IndexFlatL2(dim)  # the other index
        # index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)
        # assert not gpu_index_ivf.is_trained
        # gpu_index_ivf.train(db_embeddings) 
        # assert gpu_index_ivf.is_trained

        # gpu_index_ivf.add(db_embeddings)
        # k = args.topk               
        # D, I = gpu_index_ivf.search(embedding, k)

        # # refinement
        # st = time.time()
        # embedding = torch.tensor(embedding).unsqueeze(1).expand(N, k, dim)
        # for i in range(N):
        #     for j in range(k):
        #         if j == 0:
        #             res_j = torch.tensor(db_embeddings[I[i,j]]).unsqueeze(0)
        #         else:
        #             res_j = torch.cat([res_j, torch.tensor(db_embeddings[I[i,j]]).unsqueeze(0)], axis=0)
        #     if i == 0:
        #         refined_embedding = res_j.unsqueeze(0)
        #     else:
        #         refined_embedding = torch.cat([refined_embedding, res_j.unsqueeze(0)], axis=0)
        # cosines = cosine_similarity(embedding, refined_embedding, dim=2)
        # sorted, indices = torch.sort(cosines)
        # ranking_tensor = torch.cat(tuple([torch.index_select(torch.tensor(I[i]), 0, indices[i]).unsqueeze(0) for i in range(len(query_examples))]), 0) # in terms of numbering
        # logger.info('Time elapsed for retrieval: {:.2f} seconds'.format(time.time()-st))

        ## Using a flat index
        st = time.time()
        index_flat = faiss.IndexFlatL2(dim)  # build a flat (CPU) index
        res = faiss.StandardGpuResources()  # use a single GPU
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        k = args.topk
        gpu_index_flat.add(db_embeddings)         # add vectors to the index
        D, I = gpu_index_flat.search(embedding, k)  # actual search
        logger.info('Time elapsed for retrieval: {:.2f} seconds'.format(time.time()-st))

        # commpute mrr@k, k = 5
        # compute map
        # Evaluate on a subset of test data where the no. of ground truths >= 5
        cutoff = 0
        while cutoff < len(query_examples):
            if len(query_examples[cutoff].ground_truth) < args.topk:
                break
            cutoff += 1
        logger.info('Cutoff point for evaluation (determined by topk): {}'.format(cutoff))
        ranking_list = [[db_examples[y].index for y in x] for x in I[:cutoff]] # in terms of index
        ranking_list_2 = [[db_examples[y].text for y in x] for x in I[:cutoff]] # in terms of index
        hits = [[i+1 for i, e in enumerate(ranking_list[j]) if e in query_examples[j].ground_truth] for j in range(len(ranking_list))] # [[1, 2], [3], [3], [2], [2, 3], [2, 3]]
        mrr_score = torch.tensor([1./float(min(hit)) if len(hit) else 0.0 for hit in hits]).mean().item() # 0.52777

        is_truth = lambda x, j: 1 if (x in query_examples[j].ground_truth) else 0
        map_hits = [[is_truth(e,j) for i, e in enumerate(ranking_list[j])] for j in range(len(ranking_list))]
        hit_ratio = torch.mean(torch.mean(torch.tensor(map_hits, dtype=float), dim=1)).item()
        map_hits = torch.cumsum(torch.tensor(map_hits), dim=1)
        map_func = lambda y: torch.tensor([torch.tensor([x[i]/(i+1) for i in range(len(x))]).mean() for x in y]).mean().item() 
        map_score = map_func(map_hits) # 0.3611

        logger.info('Hit ratio: {}, MRR@{}: {}, MAP@{}: {}'.format(hit_ratio, args.topk, mrr_score, args.topk, map_score))
        res_file_path = os.path.join(args.save_dir, 'predictions.jsonl')
        logger.info('Saving predictions to {} ...'.format(res_file_path))
        for i in range(cutoff):
            query_examples[i].predictions = ranking_list[i]
        results = [{'key': query.index, 'text': query.text, 'ground_truth': query.ground_truth, 'predictions': query.predictions} for query in query_examples]
        res_file = open(res_file_path, "w")
        res_file.write(json.dumps(results, indent = 4))
        res_file.close()

        with open(os.path.join(args.save_dir, 'query_results.pkl'), 'wb') as f:
            pickle.dump(query_examples, f)
