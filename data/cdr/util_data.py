import os
import sys
import json
import torch
import logging
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
sys.path.append(rootPath)

orig_filenames = {'train':'CDR_TrainingSet.PubTator.txt', 
                  'test':'CDR_TestSet.PubTator.txt', 
                  'dev':'CDR_DevelopmentSet.PubTator.txt'}
ners = {'O':0, 'B-chem':1, 'I-chem':2, 'B-dis':3, 'I-dis':4}

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
                try:
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
                except:
                    # There is nothing we can do about it. Discard this example and move on to the next one
                    discard_flag = True
                    continue

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
                    discard_flag = False
    
    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_filename)
    output_file = open(file_dir, 'w')
    output_file.write(json.dumps(examples, indent=4))
                                                       
# if __name__ == "__main__":
#     tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
#     load_and_cache_features('test', tokenizer, 128)
