import torch
import torch.nn as nn
from adapter import Adapter
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pytorch_transformers.modeling_bert import BertAttention, BertIntermediate, BertOutput, BertLayer


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.adaptmlp = Adapter(self.config, dropout=0.1)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        residual = attention_output
        outputs = self.adaptmlp(attention_output, add_residual=False)
        outputs += layer_output

        return outputs + residual