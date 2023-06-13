from __future__ import absolute_import, division, print_function

import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, Sigmoid, MultiheadAttention, MSELoss
from torch.nn.functional import cosine_similarity
from sklearn.metrics import auc as _auc, roc_curve
from transformers import AutoModel
import json
import sys
import pandas as pd
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pytorch_transformers import RobertaModel, BertModel
from info_nce import InfoNCE

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0, min_steps=200):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.step = 0
        self.min_steps = min_steps
        self.counter = 0
        self.early_stop = False

    def __call__(self, eval, best_eval):
        if (best_eval - eval) > self.min_delta and self.step > self.min_steps:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        self.step += 1

class Example(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 index,
                 prior,
                 claim,
                 label=None):
        self.index = index
        self.prior = prior
        self.claim = claim
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.index),
            "prior: {}".format(self.prior),
            "claim: {}".format(self.claim),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return "\n".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label,
                 pseudo_label=None,
                 pseudo_fac_label=None
                 ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.pseudo_label = pseudo_label
        self.pseudo_fac_label = pseudo_fac_label


def read_examples_origin(input_file, is_training):
    examples = []
    with open(input_file) as f:
        data = json.load(f)
        for line in data:           
            index = line['index']
            prior = line['text']
            claim = line['text_b']

            examples.append(
                Example(
                    index=index,
                    prior=prior,
                    claim=claim,
                    label=line['label'] if is_training else None
                )
            )
    return examples

def read_sts_examples(input_file, is_training):
    data = pd.read_csv(input_file, sep='\t', header=None, error_bad_lines=False)
    no_examples = data.shape[0]
    examples = []
    for no in range(no_examples): 
        label, text_a, text_b = data.iloc[no, :]
        assert label is not np.nan
        examples.append(
            Example(
                index=no+1,
                prior=text_a,
                claim=text_b,
                label=label if is_training else None
            )
        )
    return examples

def tokens_to_ids(tokenizer, tokens, max_seq_length):
    # segment_ids = [0] * (len(prior) + 1) + [1] * (len(claim) + 1) + [2]
    # segment_ids = [0] * (len(context_tokens_choice) + 1) + [0] * (len(ending_tokens) + 1) + [0]
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

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 mode, logger, is_training=True, verbose=False, label_type='classification'):
    features = []
    prior_features, claim_features = [], []
    for example_index, example in enumerate(examples):
        if example_index % 1000 == 0 and verbose:
            logger.info('Processing {} examples...'.format(example_index))

        prior = tokenizer.tokenize(example.prior)
        claim = tokenizer.tokenize(example.claim)
        prior, claim = _truncate_seq_pair(prior, claim, max_seq_length, mode=mode)

        if mode == "cross":
            tokens = ['<s>'] + prior + ["</s>"] + claim + ["</s>"]
            input_ids, input_mask, segment_ids = tokens_to_ids(tokenizer, tokens, max_seq_length)
            if label_type == 'classification':
                label = int(example.label) if example.label is not None else None
            elif label_type == 'regression':
                label = float(example.label) if example.label is not None else None

            features.append(
                InputFeatures(
                    example_id=example.index,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label=label
                )
            )

        elif mode == "bi":
            prior_input_ids, prior_input_mask, prior_segment_ids = tokens_to_ids(tokenizer, ['<s>'] + prior, max_seq_length)
            claim_input_ids, claim_input_mask, claim_segment_ids = tokens_to_ids(tokenizer, ['<s>'] + claim, max_seq_length)
            if label_type == 'classification':
                label = int(example.label) if example.label is not None else None
            elif label_type == 'regression':
                label = float(example.label) if example.label is not None else None
                
            prior_features.append(
                InputFeatures(
                    example_id=example.index,
                    input_ids=prior_input_ids,
                    input_mask=prior_input_mask,
                    segment_ids=prior_segment_ids,
                    label=label
                )
            )
            claim_features.append(
                InputFeatures(
                    example_id=example.index,
                    input_ids=claim_input_ids,
                    input_mask=claim_input_mask,
                    segment_ids=claim_segment_ids,
                    label=label
                )
            )

    if mode == "cross":
        return features

    elif mode == "bi":
        return (prior_features, claim_features)


def _truncate_seq_pair(tokens_a, tokens_b, max_length, mode="cross"):
    """Truncates a sequence pair in place to the maximum length."""
    if mode == "cross":
        max_length = max_length - 3
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        
    elif mode == "bi":
        max_length -= 1
        if len(tokens_a) > max_length:
            tokens_a = tokens_a[:max_length]
        if len(tokens_b) > max_length:
            tokens_b = tokens_b[:max_length]
    
    return tokens_a, tokens_b


def sigmoid(x):
    try:
        ans = 1 / (1 + np.exp(-x))
    except:
        ans = 1 / (1 + torch.exp(-x))
    return ans

def accuracy(out, true_labels, logits=True):
    if logits:
        return np.sum((sigmoid(out) > 0.5) == true_labels)
    else:
        return np.sum((out > 0.5) == true_labels)

def auc(out, true_labels, logits=True):
    if logits:
        labels = (sigmoid(out) > 0.5)
    else:
        labels = out > 0.5
    fpr, tpr, thresholds = roc_curve(true_labels, labels, pos_label=1)
    auc_val = _auc(fpr, tpr)
    # aps = average_precision_score(true_labels, labels)
    return auc_val

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


from pytorch_transformers.modeling_bert import BertEncoder
class Adapter(nn.Module):
    def __init__(self, args,adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)
        self.mask_dtype = next(self.parameters()).dtype

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.mask_dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)
        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-self.adapter_config.initializer_range, self.adapter_config.initializer_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-self.adapter_config.initializer_range, self.adapter_config.initializer_range)


class PretrainedModel(nn.Module):
    def __init__(self, args):
        super(PretrainedModel, self).__init__()
        if args.model_name_or_path == 'bert':
            self.model = BertModel.from_pretrained("bert-base-uncased ", output_hidden_states=True)
        elif args.model_name_or_path == 'roberta-large':
            self.model = RobertaModel.from_pretrained("roberta-large", output_hidden_states=True)
        elif args.model_name_or_path == 'simcse':
            self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        elif args.model_name_or_path == 'sbert':
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        self.config = self.model.config
        self.config.freeze_adapter = args.freeze_adapter
        self.config.output_hidden_states=True
        if args.freeze_bert:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.model(input_ids=input_ids,
                             position_ids=position_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             head_mask=head_mask)
                            #  output_hidden_states=True)
                            #  return_dict=True)

        return outputs  # (logits), (pooler_outputs), (hidden_states), (attentions)
    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_pretrained_model.bin")

        torch.save(model_to_save.state_dict(), output_model_file)


class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 768
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float= 0.1
            hidden_dropout_prob: float=0.1
            hidden_size: int=768
            initializer_range: float=0.02
            intermediate_size: int=3072
            layer_norm_eps: float=1e-05
            max_position_embeddings: int=514
            num_attention_heads: int=12
            num_hidden_layers: int=self.args.adapter_transformer_layers
            num_labels: int=1
            output_attentions: bool=False
            output_hidden_states: bool=False
            torchscript: bool=False
            type_vocab_size: int=1
            vocab_size: int=50265

        self.adapter_config = AdapterConfig

        self.adapter_skip_layers = self.args.adapter_skip_layers
        self.adapter_list = args.adapter_list
        self.adapter_num = len(self.adapter_list)
        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

        if self.args.fusion_mode in ['concat', 'attention']:
            self.task_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        # if self.args.fusion_mode == 'attention':
        #     self.attention = MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=self.config.num_attention_heads)

    def init_weights(self, init_range):
        logging.info("Initialize the parameters of the taskdense")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-init_range, init_range)

    def forward(self, pretrained_model_outputs):
        outputs = pretrained_model_outputs
        sequence_output = outputs[0]
        # pooler_output = outputs[1]
        hidden_states = outputs[2]
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to('cuda')

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            # if i == 0:
            #     fusion_state = hidden_states[self.adapter_list[i]]
            # fusion_state = self.attention(hidden_states[self.adapte
            # r_list[i]], hidden_states_last, hidden_states_last)[0]
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1:
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + adapter_hidden_states[int(adapter_hidden_states_count/self.adapter_skip_layers)]

        if self.args.fusion_mode == 'add':
            task_features = self.args.a_rate * sequence_output+self.args.b_rate * hidden_states_last
        elif self.args.fusion_mode in ['concat', 'attention']:
            task_features = self.task_dense(torch.cat([self.args.a_rate * sequence_output, self.args.b_rate * hidden_states_last],dim=2))
            # task_features = self.dropout(task_features)
        # elif self.args.fusion_mode == 'attention':
        #     task_features = self.attention(sequence_output, hidden_states_last, hidden_states_last)[0]

        outputs = (task_features,) + outputs[2:]

        return outputs  # (logits), | (hidden_states), (attentions)


class patentModel(nn.Module):
    def __init__(self, args, pretrained_model_config, fac_adapter, lin_adapter, et_adapter, max_seq_length, pooling="cls", loss="bce"):
        super(patentModel, self).__init__()
        self.args = args
        self.config = pretrained_model_config

        self.fac_adapter = fac_adapter
        self.et_adapter = et_adapter
        self.lin_adapter = lin_adapter

        if args.freeze_adapter and (self.fac_adapter is not None):
            for p in self.fac_adapter.parameters():
                p.requires_grad = False
        if args.freeze_adapter and (self.lin_adapter is not None):
            for p in self.lin_adapter.parameters():
                p.requires_grad = False
        if args.freeze_adapter and (self.et_adapter is not None):
            for p in self.et_adapter.parameters():
                p.requires_grad = False
        self.adapter_num = 0
        if self.fac_adapter is not None:
            self.adapter_num += 1
        if self.lin_adapter is not None:
            self.adapter_num += 1
        if self.et_adapter is not None:
            self.adapter_num += 1

        if self.args.fusion_mode in ['concat', 'attention']:
            self.task_dense_lin = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense_fac = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense_et = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)

        if self.args.fusion_mode == 'attention':
            self.attention = MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=self.config.num_attention_heads)

        # self.num_labels = config.num_labels
        self.config = pretrained_model_config
        self.num_labels = 1
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.max_seq_length = max_seq_length
        self.pooling = pooling
        if self.pooling == "mean":
            self.classifier = nn.Linear(self.max_seq_length, 1)
        if self.pooling == "cls":
            self.classifier = nn.Linear(self.config.hidden_size, 1)
        self.loss = loss

    def forward(self, pretrained_model_outputs, labels=None, pseudo_labels=None, pseudo_fac_labels=None, labelling=False):
        batch_size = self.args.train_batch_size // self.args.gradient_accumulation_steps
        fac_adapter_outputs = torch.rand(batch_size, self.args.max_seq_length, self.config.hidden_size).to(self.args.device)
        et_adapter_outputs = torch.rand(batch_size, self.args.max_seq_length, self.config.hidden_size).to(self.args.device)
        pretrained_model_last_hidden_states = pretrained_model_outputs[0]
        if self.fac_adapter is not None:
            fac_adapter_outputs, _ = self.fac_adapter(pretrained_model_outputs)
        if self.et_adapter is not None:
            et_adapter_outputs, _ = self.et_adapter(pretrained_model_outputs)
        if self.lin_adapter is not None:
            lin_adapter_outputs, _ = self.lin_adapter(pretrained_model_outputs)

        if self.args.fusion_mode == 'add':
            task_features = pretrained_model_last_hidden_states
            if self.fac_adapter is not None:
                task_features = task_features + fac_adapter_outputs
            if self.et_adapter is not None:
                task_features = task_features + et_adapter_outputs
            if self.lin_adapter is not None:
                task_features = task_features + lin_adapter_outputs

        elif self.args.fusion_mode == 'concat':
            combine_features = pretrained_model_last_hidden_states
            if self.fac_adapter is not None:
                fac_features = self.task_dense_fac(torch.cat([combine_features, fac_adapter_outputs], dim=2))
            if self.lin_adapter is not None:
                lin_features = self.task_dense_lin(torch.cat([combine_features, lin_adapter_outputs], dim=2))
            if self.et_adapter is not None:
                et_features = self.task_dense_et(torch.cat([combine_features, et_adapter_outputs], dim=2))
            if self.fac_adapter is not None and self.lin_adapter is not None:
                task_features = self.task_dense(torch.cat([fac_features, lin_features], dim=2))
            if self.et_adapter is not None and self.lin_adapter is not None:
                task_features = self.task_dense(torch.cat([et_features, lin_features], dim=2))

        elif self.args.fusion_mode == 'attention':
            q = pretrained_model_last_hidden_states
            if self.fac_adapter is not None and self.et_adapter is not None:
                features = self.attention(et_adapter_outputs, fac_adapter_outputs, fac_adapter_outputs)[0]
                task_features = self.task_dense(torch.cat([q, features], dim=2))
            elif self.et_adapter is not None and self.lin_adapter is not None:
                features = self.attention(et_adapter_outputs, lin_adapter_outputs, lin_adapter_outputs)[0]
                task_features = self.task_dense(torch.cat([q, features], dim=2))
            elif self.fac_adapter is not None:
                features = fac_adapter_outputs
                fac_features = self.attention(q, features, features)[0]
            elif self.lin_adapter is not None:
                features = lin_adapter_outputs
                lin_features = self.attention(q, features, features)[0]
            elif self.et_adapter is not None:
                features = et_adapter_outputs
                et_features = self.attention(q, features, features)[0]
        
        if self.fac_adapter is not None and self.lin_adapter is not None:
            sequence_output = self.dropout(task_features)
        elif self.et_adapter is not None and self.lin_adapter is not None:
            sequence_output = self.dropout(task_features)
        elif self.fac_adapter is not None:
            sequence_output = self.dropout(fac_features)
        elif self.lin_adapter is not None:
            sequence_output = self.dropout(lin_features)
        else:
            sequence_output = combine_features

        sigmoid = Sigmoid()
        if labels is not None or pseudo_labels is not None or labelling:
            if self.loss == "bce":
                loss_fct = BCELoss()
            elif self.loss == "mse":
                loss_fct = MSELoss()

            if self.pooling is not None and self.pooling == "mean":
                logits = self.classifier(torch.mean(sequence_output, dim=2))
                fac_logits = self.classifier(torch.mean(fac_adapter_outputs, dim=2))

            elif self.pooling == None or self.pooling == "cls":
                logits = self.classifier(sequence_output[:, 0, :].squeeze(dim=1))
                fac_logits = self.classifier(fac_adapter_outputs[:, 0, :].squeeze(dim=1))
                
            reshaped_logits = logits.view(-1, self.num_labels)
            outputs = reshaped_logits.squeeze(dim=1)
            reshaped_fac_logits = fac_logits.view(-1, self.num_labels)
            fac_outputs = reshaped_fac_logits.squeeze(dim=1)

            if labelling:
                return sigmoid(outputs), sigmoid(fac_outputs)

            if labels is not None:
                if self.loss == "bce":
                    loss = loss_fct(sigmoid(outputs), labels) + loss_fct(sigmoid(fac_outputs), labels)
                elif self.loss == "mse":
                    outputs = self.args.score_range*sigmoid(outputs)
                    loss = loss_fct(outputs, labels) 
                    if self.args.meta_fac_adaptermodel is not '':
                        fac_outputs = self.args.score_range*sigmoid(fac_outputs)
                        loss += loss_fct(fac_outputs, labels)
            else:
                if self.loss == "bce":
                    loss = loss_fct(sigmoid(outputs), pseudo_labels) + loss_fct(sigmoid(fac_outputs), pseudo_fac_labels)
                elif self.loss == "mse":
                    loss = loss_fct(outputs, pseudo_labels) + loss_fct(fac_outputs, pseudo_fac_labels)
            return (loss, outputs, fac_outputs)
        
        elif not labels and not pseudo_labels:
            if self.args.loss == 'mse':
                sequence_output = self.args.score_range*sigmoid(sequence_output)
                fac_adapter_outputs = self.args.score_range*sigmoid(fac_adapter_outputs)
            if self.args.optimize_et_loss == True and self.et_adapter is not None:
                et_adapter_outputs = self.args.score_range*sigmoid(et_adapter_outputs)
                return sequence_output, fac_adapter_outputs, et_adapter_outputs
            return sequence_output, fac_adapter_outputs

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

def load_pretrained_adapter(adapter, adapter_path):
    new_adapter= adapter
    model_dict = new_adapter.state_dict()
    adapter_meta_dict = torch.load(adapter_path, map_location=lambda storage, loc: storage)

    for item in ['out_proj.bias', 'out_proj.weight', 'dense.weight',
                 'dense.bias']:  # 'adapter.down_project.weight','adapter.down_project.bias','adapter.up_project.weight','adapter.up_project.bias'
        if item in adapter_meta_dict:
            adapter_meta_dict.pop(item)

    changed_adapter_meta = {}
    for key in adapter_meta_dict.keys():
        changed_adapter_meta[key.replace('adapter.', 'adapter.')] = adapter_meta_dict[key]
    changed_adapter_meta = {k: v for k, v in changed_adapter_meta.items() if k in model_dict.keys()}
    model_dict.update(changed_adapter_meta)
    new_adapter.load_state_dict(model_dict)
    return new_adapter


def cosine_sim(priors, claims, normalize, device):
    bz = priors.shape[0]
    priors = torch.reshape(priors, (bz, -1))
    claims = torch.reshape(claims, (bz, -1))
    if normalize:
        priors = ((priors-torch.mean(priors))/torch.std(priors))
        claims = ((claims-torch.mean(claims))/torch.std(claims))
    cosines = cosine_similarity(priors, claims, dim=1)
    cosines = torch.where(cosines > 0, cosines, torch.zeros(bz).to(device))

    return torch.where(cosines < 1, cosines, torch.ones(bz).to(device)).unsqueeze(dim=1)

def reconstruct(prior, claim, labels, bz):
    positive = torch.where(labels == 1, claim, prior)
    for i in range(labels.shape[0]):
        if i == 0:
            if labels[i] == 0:
                negative = claim[:-1,:].unsqueeze(0)
            else:
                negative = claim[1:,:].unsqueeze(0)
        elif labels[i] == 0:
            negative = torch.cat([negative, claim[1:,:].unsqueeze(0)], dim=0)
        elif labels[i] == 1:
            idx = [j for j in range(bz) if j != i]
            negative = torch.cat([negative, claim[idx,:].unsqueeze(0)], dim=0)

    return positive, negative

def infonce(prior_vectors, claim_vectors, labels, prior_fac_vectors=None, claim_fac_vectors=None):
    if len(labels.shape) != 2:
        labels = labels.unsqueeze(1)
    bz = prior_vectors.shape[0]
    loss_fct = InfoNCE(negative_mode='paired')
    pos, neg = reconstruct(prior_vectors, claim_vectors, labels, bz)
    loss = loss_fct(prior_vectors, pos, neg)
    if prior_fac_vectors is not None and claim_fac_vectors is not None:
        fac_pos, fac_neg = reconstruct(prior_fac_vectors, claim_fac_vectors, labels, bz)
        loss += loss_fct(prior_fac_vectors, fac_pos, fac_neg)
    return loss

def load_model(model_path, model):
    changed_bert_meta = {}
    model_dict = model.state_dict()
    bert_meta_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    for key in bert_meta_dict.keys():
        if key in model_dict.keys():
            changed_bert_meta[key] = bert_meta_dict[key]
    model_dict.update(changed_bert_meta)
    model.load_state_dict(model_dict)
    return model