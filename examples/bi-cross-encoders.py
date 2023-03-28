from __future__ import absolute_import, division, print_function

import logging
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from evaluation import run_eval
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from transformers import AutoModel, AutoTokenizer
from itertools import cycle
from pytorch_transformers import AdamW, WarmupLinearSchedule

from util import read_examples_origin, convert_examples_to_features
from util import set_seed, PretrainedModel, AdapterModel, patentModel, load_pretrained_adapter, cosine_sim, load_model, sigmoid
from parser import parse

def main():
    # Parse parameters
    parser = argparse.ArgumentParser()
    args = parse(parser)
    args.output_subdir = os.path.join(args.output_dir, args.output_folder)
    try:
        os.makedirs(args.output_subdir)
    except:
        pass
    if os.path.exists(args.output_subdir) and os.listdir(args.output_subdir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_subdir))
    
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

    # Instantiate tokenizers and base models
    if args.model_name_or_path == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    elif args.model_name_or_path == 'simsce':
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
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

    patent_model = patentModel(args,pretrained_model.config,fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter, max_seq_length=args.max_seq_length, pooling="cls", loss="bce")
    pretrained_model.to(args.device)
    patent_model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Preprocess and cache train/validation data
    train_examples = read_examples_origin(os.path.join(args.data_dir, 'train.jsonl'), is_training=True)
    logger.info("Preprocessing training data for bi-encoder")
    train_features_bi = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length, "bi", logger, is_training=True)
    logger.info("Preprocessing training data for cross-encoder")
    train_features_cross = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length, "cross", logger, is_training=True)

    eval_examples = read_examples_origin(os.path.join(args.data_dir, 'valid.jsonl'), is_training=True)
    logger.info("Preprocessing training data for bi-encoder")
    eval_features_bi = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, "bi", logger, is_training=True)
    logger.info("Preprocessing training data for cross-encoder")
    eval_features_cross = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, "cross", logger, is_training=True)

    # eval_prior, eval_claim
    eval_prior_features, eval_claim_features = eval_features_bi
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
    eval_data_bi = (eval_prior_data, eval_claim_data)
    del prior_all_input_ids, prior_all_input_mask, prior_all_segment_ids, prior_all_label, eval_prior_data
    del claim_all_input_ids, claim_all_input_mask, claim_all_segment_ids, claim_all_label, eval_claim_data

    all_input_ids_cross = torch.tensor([f.input_ids for f in eval_features_cross], dtype=torch.long)
    all_input_mask_cross = torch.tensor([f.input_mask for f in eval_features_cross], dtype=torch.long)
    all_segment_ids_cross = torch.tensor([f.segment_ids for f in eval_features_cross], dtype=torch.long)
    all_label_cross = torch.tensor([f.label for f in eval_features_cross], dtype=torch.float32)
    eval_data_cross = TensorDataset(all_input_ids_cross, all_input_mask_cross, all_segment_ids_cross, all_label_cross)
    del all_input_ids_cross, all_input_mask_cross, all_segment_ids_cross, all_label_cross

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.train_steps > 0:
        num_train_optimization_steps = args.train_steps
    else:
        args.train_steps = int(args.num_train_epochs * len(train_examples) // args.train_batch_size)
        num_train_optimization_steps = args.train_steps

    bicheckpoints = ['./proc_data/roberta_patentsim_compact/cls_pooling_bce_12_epochs/']
    # bicheckpoints = ['./proc_data/roberta_patentmatch/patentmatch_batch-8_lr-5e-06_warmup-0_epoch-6.0_baseline/']
    for _cycle in range(1, args.cycles+1):
        logging.info (f"########## cycle {_cycle:.0f} starts ##########")
        logging.info ("Label sentence pairs with bi-encoder...")

        train_prior_features, train_claim_features = train_features_bi
        prior_all_input_ids = torch.tensor([f.input_ids for f in train_prior_features], dtype=torch.long).to(device)
        prior_all_input_mask = torch.tensor([f.input_mask for f in train_prior_features], dtype=torch.long).to(device)
        prior_all_segment_ids = torch.tensor([f.segment_ids for f in train_prior_features], dtype=torch.long).to(device)

        claim_all_input_ids = torch.tensor([f.input_ids for f in train_claim_features], dtype=torch.long).to(device)
        claim_all_input_mask = torch.tensor([f.input_mask for f in train_claim_features], dtype=torch.long).to(device)
        claim_all_segment_ids = torch.tensor([f.segment_ids for f in train_claim_features], dtype=torch.long).to(device)

        # Load bi-encoder checkpoint
        if _cycle == 1:
            bicheckpoint = bicheckpoints[_cycle-1]
            logger.info('Load pre-trained bert model state dict from {}'.format(bicheckpoint+'pytorch_bertmodel_best.bin'))
            pretrained_model = load_model(bicheckpoint+'pytorch_bertmodel_best.bin', pretrained_model)
            logger.info('Load pertrained patent model state dict from {}'.format(bicheckpoint+'pytorch_model_best.bin'))
            patent_model = load_model(bicheckpoint+'pytorch_model_best.bin', patent_model)
        else:
            logger.info('Load pre-trained bert model state dict from {}'.format("pytorch_model_bi_best_"+str(_cycle-1)+".bin"))
            pretrained_model = load_model(args.output_subdir + "/pytorch_model_bi_best_"+str(_cycle-1)+".bin", pretrained_model)
            logger.info('Load pertrained patent model state dict from {}'.format("pytorch_bertmodel_bi_best_"+str(_cycle-1)+".bin"))
            patent_model = load_model(args.output_subdir + "/pytorch_bertmodel_bi_best_"+str(_cycle-1)+".bin", patent_model)
        pretrained_model.eval()
        patent_model.eval()

        # Pseudo-labelling with bi-encoder
        length = len(train_prior_features)
        print('Number of examples to annotate:', length)
        with torch.no_grad():
            for i in range(length):
                prior_pretrained_model_outputs = pretrained_model(input_ids=prior_all_input_ids[i].unsqueeze(0), token_type_ids=prior_all_segment_ids[i].unsqueeze(0), attention_mask=prior_all_input_mask[i].unsqueeze(0))
                prior_output_logits, prior_fac_logits = patent_model(prior_pretrained_model_outputs)
                claim_pretrained_model_outputs = pretrained_model(input_ids=claim_all_input_ids[i].unsqueeze(0), token_type_ids=claim_all_segment_ids[i].unsqueeze(0), attention_mask=claim_all_input_mask[i].unsqueeze(0))
                claim_output_logits, claim_fac_logits = patent_model(claim_pretrained_model_outputs)
                          
                prior_vectors = prior_output_logits[:,0,:]
                claim_vectors = claim_output_logits[:,0,:]
                prior_fac_vectors = prior_fac_logits[:,0,:]
                claim_fac_vectors = claim_fac_logits[:,0,:]

                cosines = cosine_sim(prior_vectors, claim_vectors, args.normalize, device)
                fac_cosines = cosine_sim(prior_fac_vectors, claim_fac_vectors, args.normalize, device)
                train_features_cross[i].pseudo_label = cosines[0, 0]
                train_features_cross[i].pseudo_fac_label = fac_cosines[0, 0]
                if (i+1) % (length // 5) == 0:
                    print('Bi-encoder labeling: '+str(int((i+1)*100/length)) + '% completed.')
        del pretrained_model, patent_model
        print('Finsihed!')

        # Data preparation for cross-encoder training
        tb_writer = SummaryWriter(log_dir="runs/" + args.output_folder)
        all_input_ids = torch.tensor([f.input_ids for f in train_features_cross], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_cross], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_cross], dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features_cross], dtype=torch.float32)
        all_pseudo_label = torch.tensor([f.pseudo_label for f in train_features_cross], dtype=torch.float32)
        all_pseudo_fac_label = torch.tensor([f.pseudo_fac_label for f in train_features_cross], dtype=torch.float32)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_pseudo_label, all_pseudo_fac_label)
        train_sampler = RandomSampler(train_data)
        batch_size = args.train_batch_size // args.gradient_accumulation_steps_cross
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        train_dataloader = cycle(train_dataloader)
        del all_input_ids, all_input_mask, all_segment_ids, all_label, all_pseudo_label, all_pseudo_fac_label

        # Load fresh models for cross-encoder training
        pretrained_model = PretrainedModel(args)
        patent_model = patentModel(args,pretrained_model.config,fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter, max_seq_length=args.max_seq_length, pooling="cls", loss="bce")
        pretrained_model.to(args.device)
        patent_model.to(args.device)
        pretrained_model.train()
        patent_model.train()

        # Set up optimizer and scheduler
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in patent_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in patent_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=num_train_optimization_steps)
        global_step = 0

        # Train the cross-encoder
        logging.info ("Training the cross-encoder...")
        tr_loss, nb_tr_steps = 0.0, 0
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        best_score = 0
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, pseudo_labels, pseudo_fac_labels = batch

            # Model fitting
            pretrained_model_outputs = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            loss, output_logits, fac_output_logits = patent_model(pretrained_model_outputs, pseudo_labels=pseudo_labels, pseudo_fac_labels=pseudo_fac_labels)
            output_logits = output_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            if args.gradient_accumulation_steps_cross > 1:
                loss = loss / args.gradient_accumulation_steps_cross
            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps_cross / (nb_tr_steps + 1), 4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_steps += 1

            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps_cross == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if (global_step + 1) % args.eval_steps == 0 and eval_flag:
                eval_flag = False
                logger.info("***** Running evaluation *****")
                pretrained_model.eval()
                patent_model.eval()
                eval_accuracy, eval_auc = run_eval(eval_data_cross, pretrained_model, patent_model, logger, args, device, global_step, tb_writer, train_loss, batch_size, "cross", _cycle)
                if args.metrics == 'accuracy':
                    if eval_accuracy > best_score:
                        best_score = eval_accuracy
                        flag = True
                elif args.metrics == 'auc':
                    if eval_auc > best_score:
                        best_score = eval_auc
                        flag = True
                print("=" * 80)
                if args.metrics == 'accuracy':
                    print("Best Acc", best_score)
                elif args.metrics == 'auc':
                    print("Best AUC", best_score)
                if flag:
                    output_model_file = os.path.join(args.output_subdir, "pytorch_model_cross_best_"+str(_cycle)+".bin")
                    print("Saving Model to " + output_model_file + " ......")            
                    # Save a trained model
                    model_to_save = patent_model.module if hasattr(patent_model,
                                                            'module') else patent_model  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_model_file = os.path.join(args.output_subdir, "pytorch_bertmodel_cross_best_"+str(_cycle)+".bin")
                    print("Saving Model to " + output_model_file + " ......")      
                    model_to_save = pretrained_model.module if hasattr(pretrained_model,
                                                                'module') else pretrained_model
                    torch.save(model_to_save.state_dict(), output_model_file)

                pretrained_model.train()
                patent_model.train()
        
        logging.info("Label sentence pairs with cross-encoder...")
        all_input_ids = torch.tensor([f.input_ids for f in train_features_cross], dtype=torch.long).to(device)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_cross], dtype=torch.long).to(device)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_cross], dtype=torch.long).to(device)

        # Load cross-encoder checkpoint
        logger.info('Load pre-trained bert model state dict from {}'.format(args.output_subdir+"pytorch_model_cross_best_"+str(_cycle)+".bin"))
        pretrained_model = load_model(os.path.join(args.output_subdir, "pytorch_model_cross_best_"+str(_cycle)+".bin"), pretrained_model)
        logger.info('Load pertrained patent model state dict from {}'.format(args.output_subdir+"pytorch_bertmodel_cross_best_"+str(_cycle)+".bin"))
        patent_model = load_model(os.path.join(args.output_subdir, "pytorch_bertmodel_cross_best_"+str(_cycle)+".bin"), patent_model)
        pretrained_model.eval()
        patent_model.eval()

        # Pseudo-labelling with cross-encoder
        length = len(train_features_cross)
        print('Number of examples to annotate:', length)
        with torch.no_grad():
            for i in range(len(train_features_cross)):
                pretrained_model_outputs = pretrained_model(input_ids=all_input_ids[i].unsqueeze(0), token_type_ids=all_segment_ids[i].unsqueeze(0), attention_mask=all_input_mask[i].unsqueeze(0))
                output_logits, fac_logits = patent_model(pretrained_model_outputs, labelling=True)
                train_prior_features[i].pseudo_label = output_logits[0]
                train_prior_features[i].pseudo_fac_label = fac_logits[0]
                if (i+1) % (length // 5) == 0:
                    print('Cross-encoder labeling: '+str(int((i+1)*100/length)) + '% completed.')
        if args.local_rank in [-1, 0]:
            tb_writer.close()   
        del pretrained_model, patent_model, tb_writer, train_dataloader, optimizer_grouped_parameters, optimizer, scheduler
        print('Finsihed!')

        # Data preparation for bi-encoder training
        batch_size = args.train_batch_size // args.gradient_accumulation_steps_bi
        tb_writer = SummaryWriter(log_dir="runs/" + args.output_folder)
        prior_all_input_ids = torch.tensor([f.input_ids for f in train_prior_features], dtype=torch.long)
        prior_all_input_mask = torch.tensor([f.input_mask for f in train_prior_features], dtype=torch.long)
        prior_all_segment_ids = torch.tensor([f.segment_ids for f in train_prior_features], dtype=torch.long)
        prior_all_label = torch.tensor([f.label for f in train_prior_features], dtype=torch.float32)
        prior_all_pseudo_label = torch.tensor([f.pseudo_label for f in train_prior_features], dtype=torch.float32)
        prior_all_pseudo_fac_label = torch.tensor([f.pseudo_fac_label for f in train_prior_features], dtype=torch.float32)
        train_prior_data = TensorDataset(prior_all_input_ids, prior_all_input_mask, prior_all_segment_ids, prior_all_label, prior_all_pseudo_label, prior_all_pseudo_fac_label)

        claim_all_input_ids = torch.tensor([f.input_ids for f in train_claim_features], dtype=torch.long)
        claim_all_input_mask = torch.tensor([f.input_mask for f in train_claim_features], dtype=torch.long)
        claim_all_segment_ids = torch.tensor([f.segment_ids for f in train_claim_features], dtype=torch.long)
        claim_all_label = torch.tensor([f.label for f in train_prior_features], dtype=torch.float32)
        claim_all_pseudo_label = torch.tensor([f.pseudo_label for f in train_prior_features], dtype=torch.float32)
        claim_all_pseudo_fac_label = torch.tensor([f.pseudo_fac_label for f in train_prior_features], dtype=torch.float32)
        train_claim_data = TensorDataset(claim_all_input_ids, claim_all_input_mask, claim_all_segment_ids, claim_all_label, claim_all_pseudo_label, claim_all_pseudo_fac_label)

        train_prior_dataloader = DataLoader(train_prior_data, batch_size=batch_size)
        train_claim_dataloader = DataLoader(train_claim_data, batch_size=batch_size)
        train_prior_dataloader = cycle(train_prior_dataloader)
        train_claim_dataloader = cycle(train_claim_dataloader)
        del prior_all_input_ids, prior_all_input_mask, prior_all_segment_ids, prior_all_label, prior_all_pseudo_label, prior_all_pseudo_fac_label
        del claim_all_input_ids, claim_all_input_mask, claim_all_segment_ids, claim_all_label, claim_all_pseudo_label, claim_all_pseudo_fac_label

        # Load fresh models for bi-encoder training
        pretrained_model = PretrainedModel(args)
        patent_model = patentModel(args,pretrained_model.config,fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter, max_seq_length=args.max_seq_length, pooling="cls", loss="bce")
        pretrained_model.to(args.device)
        patent_model.to(args.device)
        pretrained_model.train()
        patent_model.train()

        # Set up optimizer and scheduler
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in patent_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in patent_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=num_train_optimization_steps)
        global_step = 0

        # Train the bi-encoder
        logging.info ("Training the bi-encoder...")
        tr_loss, nb_tr_steps = 0.0, 0
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        best_score = 0
        loss_fct = BCELoss()
        for step in bar:
            prior_batch = next(train_prior_dataloader)
            claim_batch = next(train_claim_dataloader)
            prior_batch = tuple(t.to(device) for t in prior_batch)
            claim_batch = tuple(t.to(device) for t in claim_batch)
            prior_input_ids, prior_input_mask, prior_segment_ids, prior_label_ids, prior_pseudo_labels, prior_pseudo_fac_labels = prior_batch
            claim_input_ids, claim_input_mask, claim_segment_ids, claim_label_ids, claim_pseudo_labels, claim_pseudo_fac_labels = claim_batch
            assert torch.ne(prior_label_ids, claim_label_ids).sum() == 0

            # Model fitting
            prior_pretrained_model_outputs = pretrained_model(input_ids=prior_input_ids, token_type_ids=prior_segment_ids, attention_mask=prior_input_mask)
            prior_output_logits, prior_fac_logits = patent_model(prior_pretrained_model_outputs)

            claim_pretrained_model_outputs = pretrained_model(input_ids=claim_input_ids, token_type_ids=claim_segment_ids, attention_mask=claim_input_mask)
            claim_output_logits, claim_fac_logits = patent_model(claim_pretrained_model_outputs)

            prior_vectors = prior_output_logits[:,0,:]
            claim_vectors = claim_output_logits[:,0,:]
            prior_fac_vectors = prior_fac_logits[:,0,:]
            claim_fac_vectors = claim_fac_logits[:,0,:]
            cosines = cosine_sim(prior_vectors, claim_vectors, args.normalize, device) # -> (bz, 1)
            fac_cosines = cosine_sim(prior_fac_vectors, claim_fac_vectors, args.normalize, device)
            labels = prior_label_ids.unsqueeze(dim=1)
            loss = loss_fct(1 / (1 + torch.exp(-cosines)), labels) + loss_fct(1 / (1 + torch.exp(-fac_cosines)), labels)
            if args.gradient_accumulation_steps_bi > 1:
                loss = loss / args.gradient_accumulation_steps_bi

            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps_bi / (nb_tr_steps + 1), 4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_steps += 1

            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps_bi == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if (global_step + 1) % args.eval_steps == 0 and eval_flag:
                eval_flag = False
                logger.info("***** Running evaluation *****")
                pretrained_model.eval()
                patent_model.eval()
                eval_accuracy, eval_auc = run_eval(eval_data_bi, pretrained_model, patent_model, logger, args, device, global_step, tb_writer, train_loss, batch_size, "bi", _cycle)
                if args.metrics == 'accuracy':
                    if eval_accuracy > best_score:
                        best_score = eval_accuracy
                        flag = True
                elif args.metrics == 'auc':
                    if eval_auc > best_score:
                        best_score = eval_auc
                        flag = True
                print("=" * 80)
                if args.metrics == 'accuracy':
                    print("Best Acc", best_score)
                elif args.metrics == 'auc':
                    print("Best AUC", best_score)
                if flag:
                    output_model_file = os.path.join(args.output_subdir, "pytorch_model_bi_best_"+str(_cycle)+".bin")
                    print("Saving Model to " + output_model_file + " ......")            
                    # Save a trained model
                    model_to_save = patent_model.module if hasattr(patent_model,
                                                            'module') else patent_model  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_model_file = os.path.join(args.output_subdir, "pytorch_bertmodel_bi_best_"+str(_cycle)+".bin")
                    print("Saving Model to " + output_model_file + " ......")      
                    model_to_save = pretrained_model.module if hasattr(pretrained_model,
                                                                'module') else pretrained_model
                    torch.save(model_to_save.state_dict(), output_model_file)
                pretrained_model.train()
                patent_model.train()

        if args.local_rank in [-1, 0]:
            tb_writer.close()    
        del pretrained_model, patent_model, tb_writer, train_prior_dataloader, train_claim_dataloader, optimizer_grouped_parameters, optimizer, scheduler
                    
if __name__ == "__main__":
    main()
