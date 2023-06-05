from __future__ import absolute_import, division, print_function

import logging
import os
import math
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
from transformers import AutoTokenizer
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

    if args.model_name_or_path == 'roberta-large':
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
        patent_model = patentModel(args, pretrained_model.config, fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter, max_seq_length=args.max_seq_length, pooling="cls", loss="bce")
        pretrained_model.to(args.device)
        patent_model.to(args.device)
        logger.info('Load pre-trained bert model state dict from {}'.format(args.pretrained_model_ckpt))
        pretrained_model = load_model(args.pretrained_model_ckpt, pretrained_model)
        logger.info('Load pertrained patent model state dict from {}'.format(args.patent_model_ckpt))
        patent_model = load_model(args.patent_model_ckpt, patent_model)
        pretrained_model.eval()
        patent_model.eval()
        model = (pretrained_model, patent_model)

    elif args.model_name_or_path == 'simcse':
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        pretrained_model = PretrainedModel(args)
        pretrained_model.to(args.device)
        logger.info('Load pre-trained bert model state dict from {}'.format(args.pretrained_model_ckpt))
        pretrained_model = load_model(args.pretrained_model_ckpt, pretrained_model)
        pretrained_model.eval()
        model = (pretrained_model, )

    if args.function == 'construct_db':
        db_examples, embedding = compute_embed_pool(args.corpus_index_file, args.corpus_content_file, tokenizer, args.save_dir, 
                                                    model, logger, device, max_seq_length=args.max_seq_length)
        with open(os.path.join(args.save_dir, 'db_examples.pkl'), 'wb') as f:
            pickle.dump(db_examples, f)
        np.save(os.path.join(args.save_dir, 'db_embedding.npy'), embedding)
        logger.info('Finished!')
    
    elif args.function == 'query':
        st = time.time()
        db_embeddings = np.load(os.path.join(args.save_dir, 'db_embedding.npy'), encoding="latin1")
        with open(os.path.join(args.save_dir, 'db_examples.pkl'), 'rb') as f:
            db_examples = pickle.load(f)
        query_examples, embedding = compute_query_pool(args.query_file, tokenizer, args.save_dir, model, logger,
                                                       device, max_seq_length=args.max_seq_length)
        N = embedding.shape[0]
        dim = embedding.shape[1]
        logger.info('Time elapsed in loading db and constructing query embeddings: {:.2f} seconds'.format(time.time()-st))

        ## Using a flat index, k = 3, 5, 10
        st = time.time()
        index_flat = faiss.IndexFlatL2(dim)  # build a flat (CPU) index
        res = faiss.StandardGpuResources()  # use a single GPU
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(db_embeddings)         # add vectors to the index
        D, I = gpu_index_flat.search(embedding, args.topk)  # actual search
        logger.info('Time elapsed for retrieval: {:.2f} seconds'.format(time.time()-st))

        ## Compute map@k
        ranking_list = [[db_examples[y].index for y in x] for x in I] # in terms of index
        is_truth = lambda x, j: 1 if (x in query_examples[j].ground_truth) else 0
        hits = [[i+1 for i, e in enumerate(ranking_list[j]) if e in query_examples[j].ground_truth] for j in range(len(ranking_list))]
        map_hits = [[is_truth(e,j) for i, e in enumerate(ranking_list[j])] for j in range(len(ranking_list))]
        map_cum_hits = torch.cumsum(torch.tensor(map_hits), dim=1)
        map_score = torch.tensor([torch.tensor([map_cum_hits[xx][i]/(i+1) if map_hits[xx][i] == 1 else 0 for i in range(len(map_cum_hits[xx]))]).sum()/len(hits[xx]) if len(hits[xx])>0 else 0 for xx in range(len(map_cum_hits))]).mean().item()

        ## Compute ndcg@k
        map_hits = np.array(map_hits)
        denom = np.broadcast_to(np.log2(np.arange(args.topk)+2), (len(map_hits), args.topk))
        dcg = np.sum(map_hits / denom, axis=1)
        idcg = np.sum(-np.sort(-map_hits, axis=1) / denom, axis=1)
        idcg = np.where(idcg > 0, idcg, math.inf)
        ndcg = np.mean(dcg / idcg)

        ## Write prediction results to output file and print evaluation outcomes
        logger.info('MAP@{}: {}, NDCG@{}: {}'.format(args.topk, map_score, args.topk, ndcg))
        res_file_path = os.path.join(args.save_dir, 'predictions.jsonl')
        logger.info('Saving predictions to {} ...'.format(res_file_path))
        for i in range(len(ranking_list)):
            query_examples[i].predictions = ranking_list[i]
        results = [{'key': query.index, 'text': query.text, 'ground_truth': query.ground_truth, 'predictions': query.predictions} for query in query_examples]
        res_file = open(res_file_path, "w")
        res_file.write(json.dumps(results, indent = 4))
        res_file.close()

        with open(os.path.join(args.save_dir, 'query_results.pkl'), 'wb') as f:
            pickle.dump(query_examples, f)
