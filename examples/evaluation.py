from torch.utils.data import DataLoader, SequentialSampler
import torch
from torch.nn import BCELoss, MSELoss
import numpy as np
import os
from util import accuracy, sigmoid, auc, cosine_sim

def run_eval(eval_data, pretrained_model, patent_model, logger, args, device, global_step, tb_writer, train_loss, batch_size, mode, _cycle):
    if type(eval_data) == tuple:
        eval_prior_data, eval_claim_data = eval_data
        eval_prior_dataloader = DataLoader(eval_prior_data, batch_size=1)
        eval_claim_dataloader = iter(DataLoader(eval_claim_data, batch_size=1))
    else:
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    if mode == "cross":
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                pretrained_model_outputs = pretrained_model(input_ids=input_ids, token_type_ids=segment_ids,
                                    attention_mask=input_mask, labels=label_ids)

                tmp_eval_loss, logits, fac_logits = patent_model(pretrained_model_outputs, labels=label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
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

    elif mode == "bi":
        loss_fct = BCELoss()
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
                prior_output_logits, prior_fac_logits = patent_model(prior_pretrained_model_outputs)

                claim_pretrained_model_outputs = pretrained_model(input_ids=claim_input_ids, token_type_ids=claim_segment_ids, attention_mask=claim_input_mask)
                claim_output_logits, claim_fac_logits = patent_model(claim_pretrained_model_outputs)

                prior_vectors = prior_output_logits[:,0,:]
                claim_vectors = claim_output_logits[:,0,:]
                prior_fac_vectors = prior_fac_logits[:,0,:]
                claim_fac_vectors = claim_fac_logits[:,0,:]
                
                cosines = cosine_sim(prior_vectors, claim_vectors, args.normalize, device) # -> (bz, 1)
                fac_cosines = cosine_sim(prior_fac_vectors, claim_fac_vectors, args.normalize, device)

                # loss = loss_fct(cosines, label_ids.unsqueeze(dim=1)) + loss_fct(fac_cosines, label_ids.unsqueeze(dim=1))
                labels = label_ids.unsqueeze(dim=1) # -> (bz, 1)
                loss = loss_fct(sigmoid(cosines), labels) + loss_fct(sigmoid(fac_cosines), labels)
                eval_loss += loss

                output_logits = cosines.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                eval_accuracy += accuracy(output_logits, label_ids, logits=False)

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
        if mode == "cross":
            eval_auc = auc(logits_auc_, label_ids_auc_)
        elif mode == "bi":
            eval_auc = auc(logits_auc_, label_ids_auc_, logits=False)

        if mode == "bi":
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

    output_eval_file = os.path.join(args.output_subdir, "eval_results_"+mode+"_"+str(_cycle)+".txt")
    with open(output_eval_file, "a", encoding='utf8') as writer:
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write('*' * 80)
        writer.write('\n')
    
    return eval_accuracy, eval_auc
        