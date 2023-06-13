from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from pytorch_transformers.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer
from itertools import cycle
from pytorch_transformers import AdamW, WarmupLinearSchedule
from util import EarlyStopping, read_examples_origin, read_sts_examples, convert_examples_to_features
from util import sigmoid, auc, set_seed, PretrainedModel, AdapterModel, patentModel, load_pretrained_adapter, cosine_sim, accuracy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: bert, roberta-large, simcse, sbert")
    parser.add_argument("--comment", default='', type=str,
                        help="The comment")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--freeze_bert", default=False, type=bool,
                        help="freeze the parameters of pretrained model.")
    parser.add_argument("--freeze_adapter", default=False, type=bool,
                        help="freeze the parameters of adapter.")
    parser.add_argument('--preprocess_type', type=str, default='', help="How to process the input")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--metrics', type=str, default='accuracy',
                        help="Metrics to determine the best model")
    parser.add_argument('--mode', type=str, default='cross',
                        help="Either one of cross-encoder (cross) or bi-coder (bi)")
    parser.add_argument('--normalize', type=bool, default=True,
                        help="Whether to apply normalization before cosine similarity measurement in the bi-encoder setting")
    parser.add_argument('--pooling', type=str, default='cls',
                        help="Pooling scheme. Input one of (mean, cls)")
    parser.add_argument('--loss', type=str, default='bce',
                        help="Loss function. Input one of (bce, infonce, mse)")
    parser.add_argument('--sim_measure', type=str, default='cosine',
                        help="Similarity measure, used only when mode = 'bi'. Input one of (cosine, linear_transform)")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    args = parser.parse_args()

    args.do_train = True
    args.do_eval = True
    args.overwrite_output_dir = True

    name_prefix = str(args.comment)
    args.my_model_name = name_prefix
    args.output_dir = os.path.join(args.output_dir, name_prefix)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        if device == torch.device("cuda"):
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 0
        # print(torch.cuda.device_count(), args.n_gpu, device)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device
    
    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    try:
        os.makedirs(args.output_dir)
    except:
        pass

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    if args.model_name_or_path == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model_name_or_path == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    elif args.model_name_or_path == 'simcse':
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    elif args.model_name_or_path == 'sbert':
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    pretrained_model = PretrainedModel(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    pretrained_model.to(args.device)
    model = (pretrained_model)

    logger.info("Training/evaluation parameters %s", args)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    read_examples_dict = {
        'read_examples_origin': read_examples_origin,
        'read_sts_examples': read_sts_examples
    }
    convert_examples_to_features_dict = {
        'read_examples_origin': convert_examples_to_features,
        'read_sts_examples': convert_examples_to_features
    }

    early_stopping = EarlyStopping()

    if args.do_train:
        # Prepare data loader
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name)
        train_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, 'train.jsonl'),
                                                                  is_training=True)
        train_features = convert_examples_to_features_dict[args.preprocess_type](
            train_examples, tokenizer, args.max_seq_length, args.mode, logger, is_training=True)
        
        if args.train_steps > 0:
            num_train_optimization_steps = args.train_steps
        else:
            args.train_steps = int(args.num_train_epochs * len(train_examples) // args.train_batch_size)
            num_train_optimization_steps = args.train_steps
        
        if args.mode == "cross":
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label = torch.tensor([f.label for f in train_features], dtype=torch.float32)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            batch_size = batch_size=args.train_batch_size // args.gradient_accumulation_steps
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                        batch_size=batch_size)
        elif args.mode == "bi":
            train_prior_features, train_claim_features = train_features
            prior_all_input_ids = torch.tensor([f.input_ids for f in train_prior_features], dtype=torch.long)
            prior_all_input_mask = torch.tensor([f.input_mask for f in train_prior_features], dtype=torch.long)
            prior_all_segment_ids = torch.tensor([f.segment_ids for f in train_prior_features], dtype=torch.long)
            prior_all_label = torch.tensor([f.label for f in train_prior_features], dtype=torch.float32)
            train_prior_data = TensorDataset(prior_all_input_ids, prior_all_input_mask, prior_all_segment_ids, prior_all_label)

            claim_all_input_ids = torch.tensor([f.input_ids for f in train_claim_features], dtype=torch.long)
            claim_all_input_mask = torch.tensor([f.input_mask for f in train_claim_features], dtype=torch.long)
            claim_all_segment_ids = torch.tensor([f.segment_ids for f in train_claim_features], dtype=torch.long)
            claim_all_label = torch.tensor([f.label for f in train_claim_features], dtype=torch.float32)
            train_claim_data = TensorDataset(claim_all_input_ids, claim_all_input_mask, claim_all_segment_ids, claim_all_label)
    
            batch_size = batch_size=args.train_batch_size // args.gradient_accumulation_steps
            train_prior_dataloader = DataLoader(train_prior_data, batch_size=batch_size)
            train_claim_dataloader = DataLoader(train_claim_data, batch_size=batch_size)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        if args.freeze_bert:
            optimizer_grouped_parameters = []
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=num_train_optimization_steps)

        global_step = 0
        hidden_size = pretrained_model.config.hidden_size
        if args.n_gpu > 1:
            pretrained_model = torch.nn.DataParallel(pretrained_model)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_acc, best_auc = 0, 0
        if args.freeze_bert:
            pretrained_model.eval()
        else:
            pretrained_model.train()
        tr_loss, logging_loss = 0.0, 0.0
        nb_tr_examples, nb_tr_steps = 0, 0

        train_accuracy = 0  # accumulative accuracy

        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        if args.mode == "cross":
            train_dataloader = cycle(train_dataloader)
        elif args.mode == "bi":
            train_prior_dataloader = cycle(train_prior_dataloader)
            train_claim_dataloader = cycle(train_claim_dataloader) 
        eval_flag = True
        if args.loss == "bce":
                loss_fct = BCELoss()
        elif args.loss == "mse":
            loss_fct = MSELoss()

        if args.pooling == "mean":
            classifier = nn.Linear(args.max_seq_length, 1).to(device)
        if args.pooling == "cls":
            classifier = nn.Linear(hidden_size, 1).to(device)

        for step in bar:
            if args.mode == "cross":
                batch = next(train_dataloader)
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                pretrained_model_outputs = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                output_logits = pretrained_model_outputs[0]

                if args.pooling is not None and args.pooling == "mean":
                    logits = classifier(torch.mean(output_logits, dim=2))

                elif args.pooling == None or args.pooling == "cls":
                    logits = classifier(output_logits[:, 0, :].squeeze(dim=1))
                    
                reshaped_logits = logits.view(-1, 1)
                output_logits = reshaped_logits.squeeze(dim=1)

                if args.loss == "bce":
                    loss = loss_fct(sigmoid(output_logits), label_ids)
                elif args.loss == "mse":
                    loss = loss_fct(output_logits, label_ids)
                output_logits = output_logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                if args.local_rank in [-1, 0]:
                    bar.set_description("loss {}".format(train_loss))

                train_accuracy += accuracy(output_logits, label_ids)
                # np.sum((sigmoid(out) > 0.5) == true_labels)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0 and (nb_tr_steps + 1) % args.gradient_accumulation_steps == 1:
                    output_logits_auc = output_logits.copy()
                    label_ids_auc = label_ids.copy()
                else:    
                    output_logits_auc = np.concatenate((output_logits_auc, output_logits), axis=0)
                    label_ids_auc = np.concatenate((label_ids_auc, label_ids), axis=0)
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

            elif args.mode == "bi":
                prior_batch = next(train_prior_dataloader)
                claim_batch = next(train_claim_dataloader)
                prior_batch = tuple(t.to(device) for t in prior_batch)
                claim_batch = tuple(t.to(device) for t in claim_batch)
                prior_input_ids, prior_input_mask, prior_segment_ids, label_ids = prior_batch
                claim_input_ids, claim_input_mask, claim_segment_ids, claim_label_ids = claim_batch
                assert torch.ne(label_ids, claim_label_ids).sum() == 0

                prior_pretrained_model_outputs = pretrained_model(input_ids=prior_input_ids, token_type_ids=prior_segment_ids, attention_mask=prior_input_mask)
                prior_output_logits = prior_pretrained_model_outputs[0]

                claim_pretrained_model_outputs = pretrained_model(input_ids=claim_input_ids, token_type_ids=claim_segment_ids, attention_mask=claim_input_mask)
                claim_output_logits = claim_pretrained_model_outputs[0]

                if args.pooling == "mean":
                    prior_vectors = torch.mean(prior_output_logits, dim=2)
                    claim_vectors = torch.mean(claim_output_logits, dim=2)
                
                elif args.pooling == "cls":
                    prior_vectors = prior_output_logits[:,0,:]
                    claim_vectors = claim_output_logits[:,0,:]
                
                if args.sim_measure == "cosine":
                    cosines = cosine_sim(prior_vectors, claim_vectors, args.normalize, device) # -> (bz, 1)

                labels = label_ids.unsqueeze(dim=1)
                if args.loss == "bce":
                    loss = loss_fct(sigmoid(cosines), labels)
                elif args.loss == "mse":
                    loss = loss_fct(cosines, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                if args.local_rank in [-1, 0]:
                    bar.set_description("loss {}".format(train_loss))

                output_logits = cosines.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                train_accuracy += accuracy(output_logits, label_ids, logits=False)
            
                if args.logging_steps > 0 and global_step % args.logging_steps == 0 and (nb_tr_steps + 1) % args.gradient_accumulation_steps == 1:
                    output_logits_auc = output_logits.copy()
                    label_ids_auc = label_ids.copy()
                else:
                    output_logits_auc = np.concatenate((output_logits_auc, output_logits), axis=0)
                    label_ids_auc = np.concatenate((label_ids_auc, label_ids), axis=0)

                nb_tr_examples += prior_input_ids.size(0)
                nb_tr_steps += 1

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), args.max_grad_norm)

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('accuracy', train_accuracy / nb_tr_examples, global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    train_auc = auc(output_logits_auc, label_ids_auc, logits=False)
                    tb_writer.add_scalar('auc', train_auc, global_step)

            if args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag:
                eval_flag = False
                for file in ['valid.jsonl']:

                    eval_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, file),
                                                                             is_training=True)
                    inference_labels = []
                    gold_labels = []
                    eval_features = convert_examples_to_features_dict[args.preprocess_type](
                        eval_examples, tokenizer, args.max_seq_length, args.mode, logger, is_training=True)

                    if args.mode == "cross":
                        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.float32)
                        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
                        # Run prediction for full data
                        eval_sampler = SequentialSampler(eval_data)
                        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    elif args.mode == "bi":
                        eval_prior_features, eval_claim_features = eval_features
                        prior_all_input_ids = torch.tensor([f.input_ids for f in eval_prior_features], dtype=torch.long)
                        prior_all_input_mask = torch.tensor([f.input_mask for f in eval_prior_features], dtype=torch.long)
                        prior_all_segment_ids = torch.tensor([f.segment_ids for f in eval_prior_features], dtype=torch.long)
                        prior_all_label = torch.tensor([f.label for f in eval_prior_features], dtype=torch.float32)
                        eval_prior_data = TensorDataset(prior_all_input_ids, prior_all_input_mask, prior_all_segment_ids, prior_all_label)

                        claim_all_input_ids = torch.tensor([f.input_ids for f in eval_claim_features], dtype=torch.long)
                        claim_all_input_mask = torch.tensor([f.input_mask for f in eval_claim_features], dtype=torch.long)
                        claim_all_segment_ids = torch.tensor([f.segment_ids for f in eval_claim_features], dtype=torch.long)
                        claim_all_label = torch.tensor([f.label for f in eval_claim_features], dtype=torch.float32)
                        eval_claim_data = TensorDataset(claim_all_input_ids, claim_all_input_mask, claim_all_segment_ids, claim_all_label)
                
                        eval_prior_dataloader = DataLoader(eval_prior_data, batch_size=1)
                        eval_claim_dataloader = iter(DataLoader(eval_claim_data, batch_size=1))


                    pretrained_model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0

                    if args.mode == "cross":
                        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                            input_ids = input_ids.to(device)
                            input_mask = input_mask.to(device)
                            segment_ids = segment_ids.to(device)
                            label_ids = label_ids.to(device)

                            with torch.no_grad():
                                pretrained_model_outputs = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                    attention_mask=input_mask, labels=label_ids)
                                logits = pretrained_model_outputs[0]

                                if args.pooling is not None and args.pooling == "mean":
                                    logits = classifier(torch.mean(logits, dim=2))
                                elif args.pooling == None or args.pooling == "cls":
                                    logits = classifier(logits[:, 0, :].squeeze(dim=1))

                                reshaped_logits = logits.view(-1, 1)
                                logits = reshaped_logits.squeeze(dim=1)
                                if args.loss == "bce":
                                    tmp_eval_loss = loss_fct(sigmoid(logits), label_ids)
                                elif args.loss == "mse":
                                    tmp_eval_loss = loss_fct(logits, label_ids)

                            logits = logits.detach().cpu().numpy()
                            label_ids = label_ids.to('cpu').numpy()
                            tmp_eval_accuracy = accuracy(logits, label_ids)
                            inference_labels.append(sigmoid(logits)>0.5)
                            gold_labels.append(label_ids)
                            eval_loss += tmp_eval_loss.item()
                            eval_accuracy += tmp_eval_accuracy
                            
                            nb_eval_steps += 1
                            nb_eval_examples += input_ids.size(0)
                            if nb_eval_steps == 1:
                                logits_auc_ = logits.copy()
                                label_ids_auc_ = label_ids.copy()
                            else:
                                logits_auc_ = np.concatenate((logits_auc_, logits), axis=0)
                                label_ids_auc_ = np.concatenate((label_ids_auc_, label_ids), axis=0)

                    elif args.mode == "bi":
                        for prior_input_ids, prior_input_mask, prior_segment_ids, label_ids in eval_prior_dataloader:
                            claim = next(eval_claim_dataloader)
                            claim = tuple(t.to(device) for t in claim)
                            claim_input_ids, claim_input_mask, claim_segment_ids, claim_label_ids = claim

                            prior_input_ids = prior_input_ids.to(device)
                            prior_input_mask = prior_input_mask.to(device)
                            prior_segment_ids = prior_segment_ids.to(device)
                            label_ids = label_ids.to(device)

                            with torch.no_grad():
                                prior_pretrained_model_outputs = pretrained_model(input_ids=prior_input_ids, token_type_ids=prior_segment_ids, attention_mask=prior_input_mask)
                                prior_output_logits = prior_pretrained_model_outputs[0]

                                claim_pretrained_model_outputs = pretrained_model(input_ids=claim_input_ids, token_type_ids=claim_segment_ids, attention_mask=claim_input_mask)
                                claim_output_logits = claim_pretrained_model_outputs[0]

                                if args.pooling == "mean":
                                    prior_vectors = torch.mean(prior_output_logits, dim=2)
                                    claim_vectors = torch.mean(claim_output_logits, dim=2)
                                
                                elif args.pooling == "cls":
                                    prior_vectors = prior_output_logits[:,0,:]
                                    claim_vectors = claim_output_logits[:,0,:]
                                
                                if args.sim_measure == "cosine":
                                    cosines = cosine_sim(prior_vectors, claim_vectors, args.normalize, device) # -> (bz, 1)

                                # loss = loss_fct(cosines, label_ids.unsqueeze(dim=1)) + loss_fct(fac_cosines, label_ids.unsqueeze(dim=1))
                                labels = label_ids.unsqueeze(dim=1) # -> (bz, 1)

                                if args.loss == "bce":
                                    loss = loss_fct(sigmoid(cosines), labels)
                                elif args.loss == "mse":
                                    loss = loss_fct(cosines, labels)

                                eval_loss += loss

                                output_logits = cosines.detach().cpu().numpy()
                                label_ids = label_ids.to('cpu').numpy()
                                eval_accuracy += accuracy(output_logits, label_ids, logits=False)

                            inference_labels.append(output_logits>0.5)
                            gold_labels.append(label_ids)
                            nb_eval_steps += 1
                            nb_eval_examples += prior_input_ids.size(0)
                            if nb_eval_steps == 1:
                                logits_auc_ = output_logits.copy()
                                label_ids_auc_ = label_ids.copy()
                            else:
                                logits_auc_ = np.concatenate((logits_auc_, output_logits), axis=0)
                                label_ids_auc_ = np.concatenate((label_ids_auc_, label_ids), axis=0)

                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    if args.local_rank in [-1, 0]:
                        logger.info("***** Running evaluation *****")
                        logger.info("  Num examples = %d", len(eval_examples))
                        logger.info("  Batch size = %d", args.eval_batch_size)
                        if args.mode == "cross":
                            eval_auc = auc(logits_auc_, label_ids_auc_)
                        elif args.mode == "bi":
                            eval_auc = auc(logits_auc_, label_ids_auc_, logits=False)

                        if args.mode == "bi":
                            eval_loss = eval_loss.item()

                        tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                        tb_writer.add_scalar('eval_accuracy', eval_accuracy, global_step)
                        tb_writer.add_scalar('eval_auc', eval_auc, global_step)
                        
                        result = {'eval_loss': eval_loss,
                                'eval_accuracy': eval_accuracy,
                                'eval_auc': eval_auc,
                                'global_step': global_step + 1,
                                'loss': train_loss}
                       
                        logger.info(result)

                        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                        with open(output_eval_file, "a", encoding='utf8') as writer:
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))
                            writer.write('*' * 80)
                            writer.write('\n')

                        if early_stopping.early_stop:
                            break

                        if args.metrics == 'accuracy':
                            if eval_accuracy > best_acc:
                                print("=" * 80)
                                print("Best Acc", eval_accuracy)
                                print("Saving Model......")
                                best_acc = eval_accuracy
                                # Save a trained model
                                model_to_save = pretrained_model.module if hasattr(pretrained_model,
                                                                                'module') else pretrained_model
                                output_model_file = os.path.join(args.output_dir, "pytorch_bertmodel_best.bin")
                                torch.save(model_to_save.state_dict(), output_model_file)
                            else:
                                print("=" * 80)
                                print("Best Acc", best_acc)

                        elif args.metrics == 'auc':
                            if eval_auc > best_auc:
                                print("=" * 80)
                                print("Best AUC", eval_auc)
                                print("Saving Model......")
                                best_auc = eval_auc
                                # Save a trained model
                                model_to_save = pretrained_model.module if hasattr(pretrained_model,
                                                                                'module') else pretrained_model
                                output_model_file = os.path.join(args.output_dir, "pytorch_bertmodel_best.bin")
                                torch.save(model_to_save.state_dict(), output_model_file)
                            else:
                                print("=" * 80)
                                print("Best AUC", best_auc)


                if args.freeze_bert:
                    pretrained_model.eval()
                else:
                    pretrained_model.train()
                
            if early_stopping.early_stop:
                break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

if __name__ == "__main__":
    main()