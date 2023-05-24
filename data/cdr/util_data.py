import os
import sys
import json
import torch
import logging
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
sys.path.append(rootPath)
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from torch.utils.data import TensorDataset

orig_filenames = {'train':'CDR_TrainingSet.PubTator.txt', 
                  'test':'CDR_TestSet.PubTator.txt', 
                  'dev':'CDR_DevelopmentSet.PubTator.txt'}
ners = {'O':0, 'B-chem':1, 'I-chem':2, 'B-dis':3, 'I-dis':4}


class InputFeatures(object):
    def __init__(self,
                 docid,
                 input_ids,
                 input_mask,
                 segment_ids,
                 labels
                 ):
        self.docid = docid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels


def build_examples(txt_filename, json_filename, tokenizer):
    '''
    Train filename: CDR_TrainingSet.PubTator.txt
    Dev filename: CDR_DevelopmentSet.PubTator.txt
    Test filename: CDR_TestSet.PubTator.txt
    '''
    examples = []
    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), txt_filename)
    input_file = open(file_dir, 'r')
    lines = input_file.readlines()
    index = section_id = pos = 0
    discard_flag = False
    for line in lines:
        if len(line.split('|')) == 3:
            if discard_flag:
                continue
            if section_id == 0:
                docid = line.split('|')[0]
                labels = []
                sentence = line.split('|')[-1]
                section_id += 1
            elif section_id == 1:
                sentence += line.split('|')[-1]
                sentence = sentence.replace('\n', ' ')
                tokens = tokenizer.tokenize(sentence)
                token_pos = (np.cumsum(np.array([0]+[len(token) for token in tokens]))-1).tolist()
                section_id += 1
        else:
            line_to_list = line.split('\t')
            if len(line_to_list) == 6:
                if discard_flag:
                    continue
                term_to_list = line_to_list[3].split()
                try:
                    new_pos = token_pos.index(int(line_to_list[1])-1, pos)
                except:
                    try:
                        new_pos = token_pos.index(int(line_to_list[1]), pos)
                    except:
                        # There is nothing we can do about it. Discard this example and move on to the next one
                        discard_flag = True
                        continue
                labels.extend(['O']*(new_pos-pos))
                pos = new_pos

                new_pos = token_pos.index(int(line_to_list[1])+len(term_to_list[0]), pos)
                if line_to_list[4] == 'Chemical':
                    labels.extend(['B-chem']*(new_pos-pos))
                    pos = new_pos
                    new_pos = token_pos.index(int(line_to_list[2]), pos)
                    labels.extend(['I-chem']*(new_pos-pos))
                    pos = new_pos
                elif line_to_list[4] == 'Disease':
                    labels.extend(['B-dis']*(new_pos-pos))
                    pos = new_pos
                    new_pos = token_pos.index(int(line_to_list[2]), pos)
                    labels.extend(['I-dis']*(new_pos-pos))
                    pos = new_pos

            elif len(line_to_list) < 4:
                section_id = 0
                pos = 0
                if not discard_flag:
                    index += 1
                    len_diff = len(tokens) - len(labels)
                    labels.extend(['O']*len_diff)
                    assert len(labels) == len(tokens)
                    examples.append({'index':index, 'docid':docid, 'sentence':sentence, 'tokens':tokens, 'labels':labels})
                else:
                    continue
    
    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_filename)
    output_file = open(file_dir, 'w')
    output_file.write(json.dumps(examples, indent=4))


def tokens_to_ids(tokenizer, tokens, max_seq_length):
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids =  input_ids + ([0] * padding_length)
    input_mask =  input_mask + ([0] * padding_length)
    segment_ids =  segment_ids + ([0] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids


def convert_examples_to_features(input_file, tokenizer, max_seq_length, verbose=False):
    features = []
    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), input_file)
    with open(file_dir) as f:
        data = json.load(f)
        for line in data:        
            docid = line['docid']
            tokens = ['<s>'] + line['tokens'][:max_seq_length-2] + ['</s>']  # <s> is the bos token and </s> is the eos token in roberta
            labels = [-100] + [ners[label] for label in line['labels'][:max_seq_length-2]]
            padding_length = max_seq_length - len(labels)
            for padding in range(padding_length):
                labels.append(-100)
            input_ids, input_mask, segment_ids=tokens_to_ids(tokenizer, tokens, max_seq_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(labels) == max_seq_length

            features.append(
                InputFeatures(
                    docid=docid,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    labels=labels
                )
            )
    if verbose:
        for i in range(3):
            print(features[i].docid)
            print(features[i].input_ids)
            print([tokenizer.convert_tokens_to_string(tokenizer._convert_id_to_token(id)) for id in features[i].input_ids])
            print(features[i].input_mask)
            print(features[i].segment_ids)
            print(features[i].labels)
            print()
    return features
    

def load_and_cache_features(args, logger, dataset_type, tokenizer):
    '''
    Dataset mode being one of 'train', 'dev', or 'test'.
    '''
    max_seq_length = args.max_seq_length
    cached_features_file = os.path.join(curPath, 'cached_{}_features'.format(dataset_type))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        build_examples(orig_filenames[dataset_type], dataset_type+'.jsonl', tokenizer)
        features = convert_examples_to_features(dataset_type+'.jsonl', tokenizer, max_seq_length)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.labels for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset
                                                       
# if __name__ == "__main__":
#     tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
#     load_and_cache_features('test', tokenizer, 128)
