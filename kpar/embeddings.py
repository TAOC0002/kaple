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

def _truncate_seq_pair(tokens_a, tokens_b, max_seq_length):
    """Truncates a sequence pair in place to the maximum length."""
    max_length = max_seq_length - 3
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    
    return tokens_a, tokens_b

def convert_to_features(examples, tokenizer, max_seq_length, logger, verbose=False):
    for example_index, example in enumerate(examples):
        if example_index % 1000 == 0 and verbose:
            logger.info('Processing {} examples...'.format(example_index))
        tokens = ['<s>']+tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        examples[example_index].input_ids, examples[example_index].input_mask, examples[example_index].segment_ids = tokens_to_ids(tokenizer, tokens, max_seq_length)
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

def get_embedding(examples, model, device):

    if len(model) == 2:
        pretrained_model, patent_model = model
    else:
        pretrained_model = model[0]

    first = True
    corpus_size = len(examples)
    print('Corpus size:', corpus_size)
    count = 0
    interval = corpus_size // 20
    with torch.no_grad():
        for e in examples:
            count += 1
            if count % interval == 0:
                print('Processed', count, '/', corpus_size, 'examples ...')
            input_ids = torch.tensor([e.input_ids], dtype=torch.long).to(device)
            input_mask = torch.tensor([e.input_mask], dtype=torch.long).to(device)
            segment_ids = torch.tensor([e.segment_ids], dtype=torch.long).to(device)

            output_logits = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            if len(model) == 2:
                output_logits, _ = patent_model(output_logits)
            else:
                output_logits = output_logits.last_hidden_state
            if first:
                embedding = output_logits[:,0,:]
                first = False
            else:
                embedding = torch.cat((embedding, output_logits[:,0,:]), 0)
        
    # save embedding
    embedding = embedding.cpu().detach().numpy().astype('float32')
    return embedding

def compute_embed_pool(corpus_index_file, corpus_content_file, tokenizer, save_dir, model, logger, device, max_seq_length=512):
    db_examples = construct_embed_pool(corpus_index_file, corpus_content_file)
    db_features = convert_to_features(db_examples, tokenizer, max_seq_length, logger)
    embedding = get_embedding(db_features, model, device)
    return db_examples, embedding

def compute_query_pool(query_file, tokenizer, save_dir, model, logger, device, max_seq_length=512):
    query_examples = construct_test_pool(query_file)
    query_features = convert_to_features(query_examples, tokenizer, max_seq_length, logger)
    embedding = get_embedding(query_features, model, device)
    return query_examples, embedding

def construct_cross_retriever_data(query_file, corpus_index_file, corpus_content_file, tokenizer, logger, max_seq_length=512):
    db_examples = construct_embed_pool(corpus_index_file, corpus_content_file)
    query_examples = construct_test_pool(query_file)
    examples = []
    for i in range(len(query_examples)):
        if (i+1) % (len(query_examples)//5) == 0:
            logger.info('Preprocessing {}% examples ...'.format(int((i+1)/len(query_examples)*100)))
        for j in range(len(db_examples)):
            ex_idx = i * len(query_examples) + j
            prior = tokenizer.tokenize(db_examples[j].text)
            claim = tokenizer.tokenize(query_examples[i].text)
            prior, claim = _truncate_seq_pair(prior, claim, max_seq_length)
            tokens = ['<s>'] + prior + ["</s>"] + claim + ["</s>"]
            input_ids, input_mask, segment_ids = tokens_to_ids(tokenizer, tokens, max_seq_length)
            examples.append(Patent(index=query_examples[i].index + '_' + db_examples[i].index, 
                                   text=query_examples[i].text + '\n' + db_examples[i].text, 
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   segment_ids=segment_ids))
    return query_examples, db_examples, examples

def cross_retriever(examples, num_query, num_db, pretrained_model, patent_model, device, logger):
    sim_scores = []
    with torch.no_grad():
        idx = 0
        query_idx = 0
        for e in examples:
            idx += 1
            input_ids = torch.tensor([e.input_ids], dtype=torch.long).to(device)
            input_mask = torch.tensor([e.input_mask], dtype=torch.long).to(device)
            segment_ids = torch.tensor([e.segment_ids], dtype=torch.long).to(device)

            model_outputs = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            output, fac_output = patent_model(model_outputs, labelling=True)
            output = output.detach().cpu().numpy().astype('float32')[0]
            if idx % num_db == 1:
                query_idx += 1
                if idx > 1:
                    sim_scores.append(sim_score)
                    if query_idx % (num_query // 20) == 0:
                        logger.info('Retrival done for {}% queries...'.format(int(query_idx / num_query * 100)))
                sim_score = [output]
            else:
                sim_score.append(output)
        sim_scores.append(sim_score)
    return sim_scores