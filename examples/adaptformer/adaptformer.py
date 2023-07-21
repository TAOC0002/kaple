# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from modules import Block
from transformers import RobertaConfig
from pytorch_transformers.modeling_roberta import RobertaEmbeddings, RobertaModel
from torch.nn import Sigmoid, MSELoss


class AdaptFormer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, args, config, score_range=5, num_classes=1, embed_dim=1024):
        super(AdaptFormer, self).__init__()
        self.args = args
        self.config = config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.score_range = score_range

        self.classifier = nn.Linear(self.num_features, num_classes)
        self.embeddings = RobertaEmbeddings(self.config)
        self.blocks = nn.ModuleList([Block(config=self.config) for _ in range(self.config.num_hidden_layers)])

        if self.args.freeze_pretrained:
            for param in self.classifier.parameters() or param in self.embeddings.parameters():
                param.requires_grad = False
            for i in range(self.config.num_hidden_layers):
                for elements in self.blocks[i].parameters():
                    elements.requires_grad = False
                for adapter_elements in self.blocks[i].adaptmlp.parameters():
                    adapter_elements.requires_grad = True

        if self.args.freeze_adapter:
            for i in range(self.config.num_hidden_layers):
                for adapter_elements in self.blocks[i].adaptmlp.parameters():
                    adapter_elements.requires_grad = False

    def forward_features(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        for i, block in enumerate(self.blocks):
            layer_outputs = block(hidden_states=hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs

        encoder_outputs = (hidden_states,)
        sequence_output = encoder_outputs[0]
        return sequence_output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        sigmoid = Sigmoid()
        loss_fct = MSELoss()
        logits = self.forward_features(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        logits = self.classifier(logits[:, 0, :].squeeze(dim=1))
        reshaped_logits = logits.view(-1, self.num_classes)
        outputs = reshaped_logits.squeeze(dim=1)
        outputs = self.score_range * sigmoid(outputs)
        loss = loss_fct(outputs, labels) 

        return (loss, outputs)