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
from embeddings import construct_cross_retriever_data, cross_retriever
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

    if args.function == 'query':
        st = time.time()
        k = args.topk
        query_examples, db_examples, examples = construct_cross_retriever_data(args.query_file, args.corpus_index_file, args.corpus_content_file, tokenizer, logger, max_seq_length=512)
        num_query = len(query_examples)
        num_db = len(db_examples)
        logger.info('Time elapsed in preprocessing: {:.2f} seconds'.format(time.time()-st))

        # st = time.time()
        # sim_scores = torch.tensor(cross_retriever(examples, num_query, num_db, pretrained_model, patent_model, device, logger), dtype=torch.float)
        # logger.info('Time elapsed in cross-encoder retrieval: {:.2f} seconds'.format(time.time()-st))
        # with open(os.path.join(args.save_dir, 'sim_simcse_scores.pkl'), 'wb') as f:
        #     pickle.dump(sim_scores, f)

        with open(os.path.join(args.save_dir, 'sim_simcse_scores.pkl'), 'rb') as f:
            sim_scores = pickle.load(f).squeeze(dim=1)
        cutoff = 0
        while cutoff < len(query_examples):
            if len(query_examples[cutoff].ground_truth) < args.topk:
                break
            cutoff += 1
        sorted_scores, indices = torch.sort(sim_scores)
        indices = indices[:cutoff,:k]
        ranking_list = [[db_examples[y.item()].index for y in x] for x in indices] # in terms of index
        ranking_list_2 = [[db_examples[y.item()].text for y in x] for x in indices] # in terms of index

        # commpute mrr@k, k = 5
        # Evaluate on all test data
        hits = [[i+1 for i, e in enumerate(ranking_list[j]) if e in query_examples[j].ground_truth] for j in range(len(ranking_list))] # [[1, 2], [3], [3], [2], [2, 3], [2, 3]]
        mrr_score = torch.tensor([1./float(min(hit)) if len(hit) else 0.0 for hit in hits]).mean().item() # 0.52777

        # compute map
        # Evaluate on a subset of test data where the no. of ground truths >= 5

        is_truth = lambda x, j: 1 if (x in query_examples[j].ground_truth) else 0
        map_hits = [[is_truth(e,j) for i, e in enumerate(ranking_list[j])] for j in range(len(ranking_list))]
        print(map_hits)
        hit_ratio = torch.mean(torch.mean(torch.tensor(map_hits, dtype=float), dim=1)).item()
        map_hits = torch.cumsum(torch.tensor(map_hits), dim=1)
        map_func = lambda y: torch.tensor([torch.tensor([x[i]/(i+1) for i in range(len(x))]).mean() for x in y]).mean().item() 
        map_score = map_func(map_hits) # 0.3611

        logger.info('Over {} examples we have Hit ratio: {}, MRR@{}: {}, MAP@{}: {}'.format(cutoff, hit_ratio, args.topk, mrr_score, args.topk, map_score))
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
