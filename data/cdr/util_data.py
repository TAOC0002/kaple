import os
import sys
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
sys.path.append(rootPath)
from pytorch_transformers.tokenization_roberta import RobertaTokenizer

'''
Train filename: CDR_TrainingSet.PubTator.txt
Dev filename: CDR_DevelopmentSet.PubTator.txt
Test filename: CDR_TestSet.PubTator.txt
'''

def build_examples(txt_filename, json_filename):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    ners = {'B-chem':0, 'I-chem':1, 'B-dis':2, 'I-dis':3, 'O':4}
    examples = []
    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), txt_filename)
    file = open(file_dir, 'r')
    lines = file.readlines()
    index = section_id = pos = count = 0
    for line in lines:
        if count > 100:
            break
        if len(line.split('|')) == 3:
            if section_id == 0:
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
                term_to_list = line_to_list[3].split()
                try:
                    new_pos = token_pos.index(int(line_to_list[1])-1, pos)
                except:
                    new_pos = token_pos.index(int(line_to_list[1]), pos)
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
                index += 1
                section_id = 0
                pos = 0
                len_diff = len(tokens) - len(labels)
                labels.extend(['O']*len_diff)
                assert len(labels) == len(tokens)
                examples.append({'index':index, 'docid':line_to_list[0], 'sentence':sentence, 'labels':labels})
        count += 1
    for example in examples:
        print(example)
        print()
    
if __name__ == "__main__":
    build_examples('CDR_TestSet.PubTator.txt', '')

# def convert_examples_to_features():

# count = 0
# # Strips the newline character
# for line in Lines:
#     count += 1
#     print("Line{}: {}".format(count, line.strip()))
