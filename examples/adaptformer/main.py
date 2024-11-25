from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
sys.path.append(rootPath)
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from transformers import AutoTokenizer, RobertaModel
from itertools import cycle
from pytorch_transformers import AdamW, WarmupLinearSchedule
from examples.util import read_sts_examples, convert_examples_to_features, set_seed
from data.sts.stsbenchmark.util_data import build_examples_and_convert_to_features
from adaptformer import AdaptFormer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--year", default=None, type=str, required=True,
                        help="Up to which year the sts datasets are constrcuted")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: roberta-large, simcse")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--comment", default='', type=str, help="The comment")

    parser.add_argument('--meta_adaptermodel', default='', type=str, help='the adapter model')
    parser.add_argument('--preprocess_type', type=str, default='read_sts_examples', help="How to process the input")

    ## Other parameters
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
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument('--freeze_adapter', action='store_true', help='whether to freeze the adapter')
    parser.add_argument('--freeze_pretrained', action='store_true', help='whether to freeze the pretrained model')

    # --meta_bertmodel="./proc_data/roberta_patentmatch/patentmatch_batch-8_lr-5e-06_warmup-0_epoch-6.0_concat/pytorch_bertmodel_4400_0.5436605821410952.bin"
    
    args = parser.parse_args()
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
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    try:
        os.makedirs(args.output_dir)
    except:
        pass

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    if args.model_name_or_path == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        dummy_model = RobertaModel.from_pretrained('roberta-large')
        # for _, p in dummy_model.named_parameters():
        #     print(_)
        config = dummy_model.config
        pretrained_model = AdaptFormer(args, config)

        for layer_no in range(24):
            pretrained_model.blocks[layer_no].attention.self.query.weight = dummy_model.encoder.layer[layer_no].attention.self.query.weight
            pretrained_model.blocks[layer_no].attention.self.query.bias = dummy_model.encoder.layer[layer_no].attention.self.query.bias
            pretrained_model.blocks[layer_no].attention.self.key.weight = dummy_model.encoder.layer[layer_no].attention.self.key.weight
            pretrained_model.blocks[layer_no].attention.self.key.bias = dummy_model.encoder.layer[layer_no].attention.self.key.bias
            pretrained_model.blocks[layer_no].attention.self.value.weight = dummy_model.encoder.layer[layer_no].attention.self.value.weight
            pretrained_model.blocks[layer_no].attention.self.value.bias = dummy_model.encoder.layer[layer_no].attention.self.value.bias
            pretrained_model.blocks[layer_no].attention.output.dense.weight = dummy_model.encoder.layer[layer_no].attention.output.dense.weight
            pretrained_model.blocks[layer_no].attention.output.dense.bias = dummy_model.encoder.layer[layer_no].attention.output.dense.bias
            pretrained_model.blocks[layer_no].attention.output.LayerNorm.weight = dummy_model.encoder.layer[layer_no].attention.output.LayerNorm.weight
            pretrained_model.blocks[layer_no].attention.output.LayerNorm.bias = dummy_model.encoder.layer[layer_no].attention.output.LayerNorm.bias
            pretrained_model.blocks[layer_no].intermediate.dense.weight = dummy_model.encoder.layer[layer_no].intermediate.dense.weight
            pretrained_model.blocks[layer_no].intermediate.dense.bias = dummy_model.encoder.layer[layer_no].intermediate.dense.biass
            pretrained_model.blocks[layer_no].output.dense.weight = dummy_model.encoder.layer[layer_no].output.dense.weight
            pretrained_model.blocks[layer_no].output.dense.bias = dummy_model.encoder.layer[layer_no].output.dense.bias
            pretrained_model.blocks[layer_no].output.LayerNorm.weight = dummy_model.encoder.layer[layer_no].output.LayerNorm.weight
            pretrained_model.blocks[layer_no].output.LayerNorm.bias = dummy_model.encoder.layer[layer_no].output.LayerNorm.bias


    elif args.model_name_or_path == 'simcse':
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

    model_dict = pretrained_model.state_dict()
    adapter_params = [n for n, p in pretrained_model.named_parameters() if 'adaptmlp' in n]
    pretrained_params = [n for n, p in pretrained_model.named_parameters() if 'adaptmlp' not in n]
    for n, p in pretrained_model.named_parameters():
        print(n)

    read_examples_dict = {
        'read_sts_examples': read_sts_examples
    }
    convert_examples_to_features_dict = {
        'read_sts_examples': convert_examples_to_features
    }

    if args.meta_adaptermodel:
        model_dict = pretrained_model.state_dict()
        bert_meta_dict = torch.load(args.meta_adaptermodel, map_location=lambda storage, loc: storage)

        changed_bert_meta = {}
        for key in adapter_params:
            if key in model_dict.keys():
                changed_bert_meta[key] = bert_meta_dict[key]

        print('Parameters to change:', changed_bert_meta.keys())
        model_dict.update(changed_bert_meta)
        pretrained_model.load_state_dict(model_dict)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    pretrained_model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    if args.do_train:
        train_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, args.year + '.train.tsv'),
                                                                  is_training=True)
        train_features = convert_examples_to_features_dict[args.preprocess_type](
            train_examples, tokenizer, args.max_seq_length, 'cross', logger, is_training=True, label_type='regression')
        num_train_optimization_steps = int(args.num_train_epochs * len(train_features) // args.train_batch_size)

        batch_size = args.train_batch_size // args.gradient_accumulation_steps
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.float32)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                    batch_size=batch_size)
        if args.do_eval:
            eval_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, args.year + '.val.tsv'),
                                                                             is_training=True)
            eval_features = convert_examples_to_features_dict[args.preprocess_type](
                eval_examples, tokenizer, args.max_seq_length, 'cross', logger, is_training=True, label_type='regression')
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features], dtype=torch.float32)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        frozen_params = []
        if args.freeze_adapter:
            frozen_params.extend(adapter_params)
        elif args.freeze_pretrained:
            frozen_params.extend(pretrained_params)
            
        optimizer_grouped_parameters = [
                {'params': [p for n, p in pretrained_model.named_parameters() if not any(nd in n for nd in no_decay) and n not in frozen_params],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in pretrained_model.named_parameters() if any(nd in n for nd in no_decay) and n not in frozen_params],
                 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0,
                                         t_total=num_train_optimization_steps)

        global_step = 0
        if args.n_gpu > 1:
            if args.freeze_bert:
                pretrained_model = torch.nn.DataParallel(pretrained_model)

        best_spearmanr, best_personr = 0, 0
        pretrained_model.train()
        tr_loss, logging_loss = 0.0, 0.0
        nb_tr_examples, nb_tr_steps = 0, 0

        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True

        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, output_logits = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            output_logits = output_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), 1.0)

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag:
                eval_flag = False
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

                pretrained_model.eval()
                eval_loss = 0
                nb_eval_steps, nb_eval_examples = 0, 0

                for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids,
                                            attention_mask=input_mask, labels=label_ids)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    eval_loss += tmp_eval_loss.item()
                    
                    nb_eval_steps += 1
                    nb_eval_examples += input_ids.size(0)
                    if nb_eval_steps == 1:
                        inference = logits.copy()
                        gold = label_ids.copy()
                    else:
                        inference = np.concatenate((inference, logits), axis=0)
                        gold = np.concatenate((gold, label_ids), axis=0)

                eval_loss = eval_loss / nb_eval_steps

                if args.local_rank in [-1, 0]:
                    logger.info("***** Running training *****")
                    logger.info("  Num examples = %d", len(train_features))
                    logger.info("  Batch size = %d", args.train_batch_size)
                    logger.info("  Num steps = %d", num_train_optimization_steps)

                    eval_spearmanr = spearmanr(np.squeeze(inference), gold)[0]
                    eval_pearsonr = pearsonr(np.squeeze(inference), gold)[0]
                    
                    result = {'eval_loss': eval_loss,
                            'eval_spearmanr': eval_spearmanr,
                            'eval_pearsonr': eval_pearsonr,
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
                    if eval_spearmanr > best_spearmanr:
                        print("=" * 80)
                        print("Best Spearman score", eval_spearmanr)
                        print("Saving Model......")
                        best_spearmanr = eval_spearmanr
                        # Save a trained model
                        model_to_save = pretrained_model.module if hasattr(pretrained_model,
                                                                        'module') else pretrained_model
                        output_model_file = os.path.join(args.output_dir, "pytorch_bertmodel_best.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    else:
                        print("=" * 80)
                        print("Best Spearman score", best_spearmanr)

                pretrained_model.train()

    if args.do_test:
        pretrained_model.eval()
        print('test...')
        for file in [args.year + '.test.tsv']:
            eval_examples = read_examples_dict[args.preprocess_type](os.path.join(args.data_dir, file),
                                                                        is_training=True)
            eval_features = convert_examples_to_features_dict[args.preprocess_type](
                eval_examples, tokenizer, args.max_seq_length, 'cross', logger, is_training=True, label_type='regression')
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features], dtype=torch.float32)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            pretrained_model.eval()
            eval_loss = 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    loss, logits = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids,
                                        attention_mask=input_mask, labels=label_ids)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                eval_loss += tmp_eval_loss.item()
                
                nb_eval_steps += 1
                nb_eval_examples += input_ids.size(0)
                if nb_eval_steps == 1:
                    inference = logits.copy()
                    gold = label_ids.copy()
                else:
                    inference = np.concatenate((inference, logits), axis=0)
                    gold = np.concatenate((gold, label_ids), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            if args.local_rank in [-1, 0]:
                eval_spearmanr = spearmanr(np.squeeze(inference), gold)[0]
                eval_pearsonr = pearsonr(np.squeeze(inference), gold)[0]
                        
                result = {'eval_loss': eval_loss,
                        'eval_spearmanr': eval_spearmanr,
                        'eval_pearsonr': eval_pearsonr}
                
                logger.info(result)

if __name__ == "__main__":
    main()