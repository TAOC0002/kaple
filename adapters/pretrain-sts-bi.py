# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Pre-train Factual Adapter
"""
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from itertools import cycle
import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.nn import MSELoss, MultiheadAttention
import sys, os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from data.sts.stsbenchmark.util_data import build_examples_and_convert_to_features
from pytorch_transformers import (RobertaTokenizer,
                                  RobertaModel)
from pytorch_transformers import AdamW, WarmupLinearSchedule
from examples.util import sigmoid, cosine_sim

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import time
from utils_glue import processors, output_modes

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_features(args, logger, dataset_type, tokenizer):
    '''
    Dataset mode being one of 'train', 'val', or 'test'.
    '''
    max_seq_length = args.max_seq_length
    filename = 'cached_{}_{}_{}_{}'.format(
        dataset_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name)
    )
    filename_a = filename+'a'
    filename_b = filename+'b'
    cached_features_file_a = os.path.join(args.data_dir, filename_a)
    cached_features_file_b = os.path.join(args.data_dir, filename_b)
    if os.path.exists(cached_features_file_a):
        logger.info("Loading features from cached file %s", cached_features_file_a)
        features_a = torch.load(cached_features_file_a)
        features_b = torch.load(cached_features_file_b)
    else:
        features_a, features_b = build_examples_and_convert_to_features(dataset_type+'.tsv', 'bi', max_seq_length, tokenizer)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s and %s", cached_features_file_a, cached_features_file_b)
            torch.save(features_a, cached_features_file_a)
            torch.save(features_b, cached_features_file_b)
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids_a = torch.tensor([f.input_ids for f in features_a], dtype=torch.long)
    all_input_mask_a = torch.tensor([f.input_mask for f in features_a], dtype=torch.long)
    all_segment_ids_a = torch.tensor([f.segment_ids for f in features_a], dtype=torch.long)
    all_label_ids_a = torch.tensor([f.label for f in features_a], dtype=torch.float32)
    dataset_a = TensorDataset(all_input_ids_a, all_input_mask_a, all_segment_ids_a, all_label_ids_a)

    all_input_ids_b = torch.tensor([f.input_ids for f in features_b], dtype=torch.long)
    all_input_mask_b = torch.tensor([f.input_mask for f in features_b], dtype=torch.long)
    all_segment_ids_b = torch.tensor([f.segment_ids for f in features_b], dtype=torch.long)
    all_label_ids_b = torch.tensor([f.label for f in features_b], dtype=torch.float32)
    dataset_b = TensorDataset(all_input_ids_b, all_input_mask_b, all_segment_ids_b, all_label_ids_b)

    return (dataset_a, dataset_b)


def train(args, train_dataset, val_dataset, model, tokenizer):
    """ Train the model """
    loss_fct = MSELoss()
    pretrained_model = model[0]
    adapter_model = model[1]
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset_a, train_dataset_b = train_dataset
    train_sampler_a = RandomSampler(train_dataset_a) if args.local_rank == -1 else DistributedSampler(train_dataset_a)
    train_sampler_b = RandomSampler(train_dataset_b) if args.local_rank == -1 else DistributedSampler(train_dataset_b)
    train_dataloader_a = DataLoader(train_dataset_a, sampler=train_sampler_a, batch_size=args.train_batch_size)
    train_dataloader_b = DataLoader(train_dataset_b, sampler=train_sampler_b, batch_size=args.train_batch_size)
    train_dataloader_a = cycle(train_dataloader_a)
    train_dataloader_b = cycle(train_dataloader_b)

    if args.max_steps > 0:
        num_train_optimization_steps = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataset_a) // args.gradient_accumulation_steps) + 1
    else:
        args.train_steps = int(args.num_train_epochs * len(train_dataset_a) // args.train_batch_size)
        num_train_optimization_steps = args.train_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in adapter_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in adapter_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        adapter_model = torch.nn.DataParallel(adapter_model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d",
                len(train_dataset_a))  # logging.info(f"  Num train_examples = {len(train_examples)}")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_train_optimization_steps)

    logger.info("Try resume from checkpoint")

    global_step = 0
    start_epoch = 0
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name, purge_step=global_step)
    logger.info("Start from scratch")

    tr_loss, logging_loss = 0.0, 0.0
    pretrained_model.zero_grad()
    adapter_model.zero_grad()

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    best_eval_spearmanr = 0
    best_step = 0
    best_checkpoint_dir = os.path.join(args.output_dir, 'best-checkpoint')
    if not os.path.exists(best_checkpoint_dir):
        os.makedirs(best_checkpoint_dir)

    pretrained_model.eval()
    adapter_model.train()

    bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
    for step in bar:
        input_ids_a, input_mask_a, segment_ids_a, labels = next(train_dataloader_a)
        batch_b = next(train_dataloader_b)
        batch_b = tuple(t.to(args.device) for t in batch_b)
        start = time.time()

        input_ids_a, input_mask_a, segment_ids_a, labels = tuple(t.to(args.device) for t in [input_ids_a, input_mask_a, segment_ids_a, labels])
        inputs_a = {'input_ids': input_ids_a,
                    'attention_mask': input_mask_a,
                    'segment_ids': segment_ids_a if args.model_type in ['bert', 'xlnet'] else None,
                    'labels': labels}
        pretrained_model_outputs = pretrained_model(**inputs_a)
        outputs_a = adapter_model(pretrained_model_outputs)

        batch_b = tuple(t.to(args.device) for t in batch_b)
        inputs_b = {'input_ids': batch_b[0],
                    'attention_mask': batch_b[1],
                    'segment_ids': batch_b[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'labels': labels}
        pretrained_model_outputs = pretrained_model(**inputs_b)
        outputs_b = adapter_model(pretrained_model_outputs)

        cosines = cosine_sim(outputs_a, outputs_b, args.normalize, args.device)
        loss = loss_fct(args.score_range*sigmoid(cosines), labels)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), args.max_grad_norm)
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            pretrained_model.zero_grad()
            adapter_model.zero_grad()
            global_step += 1
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss
                if global_step % (args.logging_steps) == 0:
                    logger.info("----------------- Iter {} / {}, loss = {:.5f}, time used = {:.3f}s -----------------".format(step,
                     num_train_optimization_steps, loss.item(), time.time() - start))

            if args.local_rank == -1 and args.evaluate_during_training and global_step % args.eval_steps == 0:  # Only evaluate when single GPU otherwise metrics may not average well
                model = (pretrained_model,adapter_model)
                results = evaluate(args, val_dataset, model)
                for key, value in results.items():
                    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                if results['eval_spearmanr'] > best_eval_spearmanr:
                    logger.info("Saving best model checkpoint and optimizer to %s", best_checkpoint_dir)
                    best_eval_spearmanr = results['eval_spearmanr'] 
                    best_step = global_step                  
                    model_to_save = adapter_model.module if hasattr(adapter_model,
                                                            'module') else adapter_model  # Take care of distributed/parallel training
                    
                    model_to_save.save_pretrained(best_checkpoint_dir)  # save to pytorch_model.bin  model.state_dict()
                    torch.save(optimizer.state_dict(), os.path.join(best_checkpoint_dir, 'optimizer.bin'))
                    torch.save(scheduler.state_dict(), os.path.join(best_checkpoint_dir, 'scheduler.bin'))
                    torch.save(args, os.path.join(best_checkpoint_dir, 'training_args.bin'))
                logger.info("eval_spearmanr at current step: {}".format(results['eval_spearmanr']))
                logger.info("Best eval_spearmanr occurred at step {}: {:.4f}".format(best_step, best_eval_spearmanr))

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, val_dataset, model):
    pretrained_model = model[0]
    adapter_model = model[1]
    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    val_dataset_a, val_dataset_b = val_dataset
    val_sampler_a = SequentialSampler(val_dataset_a) if args.local_rank == -1 else DistributedSampler(val_dataset_a)
    val_sampler_b = SequentialSampler(val_dataset_b) if args.local_rank == -1 else DistributedSampler(val_dataset_b)
    val_dataloader_a = DataLoader(val_dataset_a, sampler=val_sampler_a, batch_size=args.eval_batch_size)
    val_dataloader_b = iter(DataLoader(val_dataset_b, sampler=val_sampler_b, batch_size=args.eval_batch_size))

    eval_loss = 0.0
    nb_eval_steps = 0
    start = time.time()
    loss_fct = MSELoss()

    for input_ids_a, input_mask_a, segment_ids_a, labels in val_dataloader_a:
        batch_b = next(val_dataloader_b)
        batch_b = tuple(t.to(args.device) for t in batch_b)

        pretrained_model.eval()
        adapter_model.eval()
        with torch.no_grad():
            input_ids_a, input_mask_a, segment_ids_a, labels = tuple(t.to(args.device) for t in [input_ids_a, input_mask_a, segment_ids_a, labels])
            inputs_a = {'input_ids': input_ids_a,
                      'attention_mask': input_mask_a,
                      'segment_ids': segment_ids_a if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': labels}
            pretrained_model_outputs = pretrained_model(**inputs_a)
            outputs_a = adapter_model(pretrained_model_outputs)

            batch_b = tuple(t.to(args.device) for t in batch_b)
            inputs_b = {'input_ids': batch_b[0],
                      'attention_mask': batch_b[1],
                      'segment_ids': batch_b[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': labels}
            pretrained_model_outputs = pretrained_model(**inputs_b)
            outputs_b = adapter_model(pretrained_model_outputs)

            cosines = cosine_sim(outputs_a, outputs_b, args.normalize, args.device)
            cosines = args.score_range*sigmoid(cosines)
            labels = labels.unsqueeze(dim=1)
            tmp_eval_loss = loss_fct(cosines, labels)

            eval_loss += tmp_eval_loss
            cosines = cosines.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()

        nb_eval_steps += 1
        if nb_eval_steps == 1:
            inference = cosines.copy()
            gold = labels.copy()
        else:
            inference = np.concatenate((inference, cosines), axis=0)
            gold = np.concatenate((gold, labels), axis=0)

    eval_loss = (eval_loss / nb_eval_steps).item()
    if args.local_rank in [-1, 0]:
        eval_spearmanr = spearmanr(np.squeeze(inference), np.squeeze(gold))[0]
        eval_pearsonr = pearsonr(np.squeeze(inference), np.squeeze(gold))[0]              
        results = {'eval_loss': eval_loss,
                'eval_spearmanr': eval_spearmanr,
                'eval_pearsonr': eval_pearsonr,
                'time_elapsed': time.time()-start}
        logger.info(results)

        output_eval_file = os.path.join(args.output_dir, args.my_model_name + "_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results  *****")
            for key in sorted(results.keys()):
                writer.write("%s = %s\n" % (key, str(results[key])))
        return results


'''
Adapter model
'''
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
        self.init_weights()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
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
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

class PretrainedModel(nn.Module):
    def __init__(self):
        super(PretrainedModel, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-large", output_hidden_states=True)
        self.config = self.model.config
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, input_ids, attention_mask=None, segment_ids=None, position_ids=None, head_mask=None, labels=None):

        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=segment_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)

        return outputs  # (loss), logits, (hidden_states), (attentions)

class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config, n_ner=1):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = self.args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
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
            num_labels: int=2
            output_attentions: bool=False
            output_hidden_states: bool=False
            torchscript: bool=False
            type_vocab_size: int=1
            vocab_size: int=50265

        self.adapter_skip_layers = self.args.adapter_skip_layers
        self.num_labels = n_ner
        # self.config.output_hidden_states=True
        self.adapter_list = args.adapter_list
        # self.adapter_list =[int(i) for i in self.adapter_list]
        self.adapter_num = len(self.adapter_list)
        # self.adapter = Adapter(args, AdapterConfig)

        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])
        self.attention = nn.ModuleList([MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=self.config.num_attention_heads) for _ in range(self.adapter_num)])
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, pretrained_model_outputs, labels=None):

        outputs = pretrained_model_outputs
        sequence_output = outputs[0]
        # pooler_output = outputs[1]
        hidden_states = outputs[2]
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(self.args.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            if i == 0:
                fusion_state = hidden_states[self.adapter_list[i]]
            else:
                fusion_state = self.attention[i](hidden_states[self.adapter_list[i]], hidden_states_last, hidden_states_last)[0]

            # fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1: # if adapter_skip_layers>=1, skip connection
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + adapter_hidden_states[int(adapter_hidden_states_count/self.adapter_skip_layers)]

        ##### drop below parameters when doing downstream tasks
        com_features = self.attention[0](sequence_output, hidden_states_last, hidden_states_last)[0]
        com_features = self.dropout(com_features)   
        logits = self.out_proj(self.dropout(com_features))

        if labels is not None:
            loss_fct = MSELoss()
            logits = sigmoid(logits)*self.args.score_range
            label_clean = labels.view(-1)
            loss = loss_fct(logits.view(-1), label_clean)
            outputs = (loss, logits)
            return outputs
        else:
            return logits

    def save_pretrained(self, save_directory):
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--score_range", default=5, type=int, help="Range of the STS scores.")
    parser.add_argument("--model_type", default='roberta', type=str, required=True,
                        help="Model type selected in the list")
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--comment", default='', type=str,
                        help="The comment")
    parser.add_argument('--output_dir', type=Path, default="output")

    parser.add_argument("--restore", type=bool, default=True,
                        help="Whether restore from the last checkpoint, is nochenckpoints, start from scartch")

    parser.add_argument("--max_seq_length", type=int, default=64, help="max lenght of token sequence")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", type=bool, default=False,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=128, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,23", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=6, type=int,
                        help="The skip_layers of adapter according to bert layers")
    parser.add_argument('--meta_adapter_model', type=str, help='the pretrained adapter model')

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=None,
                        help="eval every X updates steps.")
    parser.add_argument('--max_save_checkpoints', type=int, default=500,
                        help="The max amounts of checkpoint saving. Bigger than it will delete the former checkpoints")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--normalize', type=bool, default=True,
                        help="Whether to apply normalization before cosine similarity measurement in the bi-encoder setting")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # args
    args = parser.parse_args()

    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]
    args.my_model_name = args.comment
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

    if args.eval_steps is None:
        args.eval_steps = args.save_steps * 10

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    pretrained_model = PretrainedModel()
    adapter_model = AdapterModel(args, pretrained_model.config)
    model = (pretrained_model, adapter_model)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    pretrained_model.to(args.device)
    adapter_model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    val_dataset = load_and_cache_features(args, logger, 'val', tokenizer)
    if args.do_train:
        train_dataset = load_and_cache_features(args, logger, 'train', tokenizer)
        # train_dataset = RelDataset(examples=train_examples, max_seq_length=args.max_seq_length)
        global_step, tr_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

if __name__ == '__main__':
    main()
