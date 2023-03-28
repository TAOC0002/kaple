import numpy as np
import pandas as pd
import pickle
import json
import torch
import os
import itertools
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from examples.util import set_seed, PretrainedModel, AdapterModel, patentModel, load_pretrained_adapter, cosine_sim, load_model, tokens_to_ids

class Patent(object):
    def __init__(self,
                 index,
                 text,
                 input_ids=None,
                 input_mask=None,
                 segment_ids=None,
                 ground_truth=None,
                 predictions=None):
        self.index = index
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.ground_truth = ground_truth
        self.predictions = predictions

def convert_to_features(examples, tokenizer, max_seq_length, logger, verbose=False):
    for example_index, example in enumerate(examples):
        if example_index % 1000 == 0 and verbose:
            logger.info('Processing {} examples...'.format(example_index))
        tokens = tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        examples[example_index].input_ids, examples[example_index].input_mask, examples[example_index].segment_ids = tokens_to_ids(tokenizer, ['<s>']+tokens, max_seq_length)
    return examples

def construct_embed_pool(corpus_index_file, corpus_content_file, verbose=False):
    with open(corpus_index_file, 'rb') as fp:
        corpus_index_list = pickle.load(fp)
    corpus_abstract = np.load(corpus_content_file, encoding="latin1", allow_pickle=True).item()
    examples = []
    for index in corpus_index_list:
        examples.append(Patent(index, corpus_abstract[index]))
    return examples

def construct_test_pool(test_filename):
    examples = []
    with open(test_filename) as f:
        data = json.load(f)
        for line in data:           
            index = line['key']
            text = line['text']
            ground_truth = line['ground_truth']
            examples.append(Patent(index=index, text=text, ground_truth=ground_truth))
    return examples

def get_embedding(examples, pretrained_model, patent_model, device):
    first = True
    with torch.no_grad():
        for e in examples:
            input_ids = torch.tensor([e.input_ids], dtype=torch.long).to(device)
            input_mask = torch.tensor([e.input_mask], dtype=torch.long).to(device)
            segment_ids = torch.tensor([e.segment_ids], dtype=torch.long).to(device)

            model_outputs = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            output_logits, _ = patent_model(model_outputs)
            if first:
                embedding = output_logits[:,0,:]
                first = False
            else:
                embedding = torch.cat((embedding, output_logits[:,0,:]), 0)
        
    # save embedding
    embedding = embedding.cpu().detach().numpy().astype('float32')
    return embedding

def compute_embed_pool(corpus_index_file, corpus_content_file, tokenizer, save_dir, pretrained_model, patent_model, logger, device, max_seq_length=512):
    db_examples = construct_embed_pool(corpus_index_file, corpus_content_file)
    db_features = convert_to_features(db_examples, tokenizer, max_seq_length, logger)
    embedding = get_embedding(db_features, pretrained_model, patent_model, device)
    return db_examples, embedding

def compute_query_pool(query_file, tokenizer, save_dir, pretrained_model, patent_model, logger, device, max_seq_length=512):
    query_examples = construct_test_pool(query_file)
    query_features = convert_to_features(query_examples, tokenizer, max_seq_length, logger)
    embedding = get_embedding(query_features, pretrained_model, patent_model, device)
    return query_examples, embedding