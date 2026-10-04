import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from sympy import divisors

def get_small_factors(n):
    threshold = 16
    factors = divisors(n)
    small_factors = [factor for factor in factors if factor <= threshold]
    return max(small_factors)

PAD_ID = 1e-6


class Config:

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    e_n_layer = 2

    def __init__(self, action_dim=None, block_size=None, n_embd=None, decoder_num=None, **kwargs):
        self.action_dim = action_dim
        self.block_size = block_size
        self.n_embd = n_embd
        self.d_n_layer = decoder_num
        self.n_head = get_small_factors(n_embd)
        for k, v in kwargs.items():
            setattr(self, k, v)


class SelfAttention(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.device = device
        # key, query, value projection for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, layer_past=None, padding_mask=None):
        batch_size, seq_len, issue_num, embed_dim = x.size()

        k = (
            self.key(x).view(batch_size, seq_len, issue_num, self.n_head, embed_dim // self.n_head).transpose(1, 3)
        )  
        q = (
            self.query(x).view(batch_size, seq_len, issue_num, self.n_head, embed_dim // self.n_head).transpose(1, 3)
        )  
        v = (
            self.value(x).view(batch_size, seq_len, issue_num, self.n_head, embed_dim // self.n_head).transpose(1, 3)
        )  

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        attention_weight = att
        att = self.attn_drop(att)
        y = att @ v
        y = (
            y.transpose(1, 3).contiguous().view(batch_size, seq_len, issue_num, embed_dim)
        )
        y = self.resid_drop(self.proj(y))
        return y, attention_weight

class CausalAttention(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.device = device
        # key, query, value projection for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, memory=None, layer_past=None, padding_mask=None):
        if memory is None:
            memory = x
        batch_size_x, seq_len_x, issue_num_x, embed_dim_x = x.size()
        batch_size_memory, seq_len_memory, issue_num_memory, embed_dim_memory = memory.size()

        k = (
            self.key(memory).view(batch_size_memory, seq_len_memory, issue_num_memory, self.n_head, embed_dim_memory // self.n_head).transpose(1, 3)
        )  
        q = (
            self.query(x).view(batch_size_x, seq_len_x, issue_num_x, self.n_head, embed_dim_x // self.n_head).transpose(1, 3)
        )  
        v = (
            self.value(memory).view(batch_size_memory, seq_len_memory, issue_num_memory, self.n_head, embed_dim_memory // self.n_head).transpose(1, 3)
        )  

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if padding_mask is not None:
            att = att.masked_fill(padding_mask, float("-inf"))
            att = att.masked_fill(self.mask[:, :, :, :seq_len_x, :seq_len_memory] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        attention_weight = att
        att = self.attn_drop(att)
        y = att @ v
        y = (
            y.transpose(1, 3).contiguous().view(batch_size_x, seq_len_x, issue_num_x, embed_dim_x)
        )
        y = self.resid_drop(self.proj(y))
        return y, attention_weight

class Encoder(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config, device)
        self.n_head = config.n_head
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.att_weight = None

    def forward(self, x):
        x = F.normalize(x, eps=1e-8)
        att_output, att_weight = self.attn(self.ln1(x))
        self.att_weight = att_weight
        x = x + att_output
        x = x + self.mlp(self.ln2(x))
        return x
    
class Decoder(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.s_attn = CausalAttention(config, device)
        self.m_attn = CausalAttention(config, device)
        self.n_head = config.n_head
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.att_weight = None
    
    def forward(self, inputs):
        x, memory, padding_mask = inputs
        x = F.normalize(x, eps=1e-8)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input tensor contains NaN or Inf values.")
        att_output, att_weight = self.s_attn(self.ln1(x),padding_mask=padding_mask)
        self.att_weight = att_weight
        x = x + att_output
        att_output, _ = self.m_attn(
            self.ln2(x), memory, padding_mask=padding_mask if memory is None else None
        )
        x = x + att_output
        x = x + self.mlp(self.ln3(x))
        return x


class NegTransformer(nn.Module):

    def __init__(self, config, memory, device=None, inputs_embeds=True, isLearning=True, decoder_only=False):
        super().__init__()

        self.inputs_embeds = inputs_embeds
        self.isLearning = isLearning
        self.isEncoded = False
        self.device = device

        self.pos_emb_e = nn.Parameter(torch.zeros(1, config.block_size, 1, config.n_embd))
        self.pos_emb_d = nn.Parameter(torch.zeros(1, config.block_size, 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.encoder_layers = nn.Sequential(*[Encoder(config, self.device) for _ in range(config.e_n_layer)])
        self.ln_e = nn.LayerNorm(config.n_embd)

        self.decoder_layers = nn.ModuleList([Decoder(config, self.device) for _ in range(config.d_n_layer)])
        self.d_n_layer = config.d_n_layer
        self.ln_d = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(config.n_embd, config.n_embd)
        self.ln_h = nn.LayerNorm(config.n_embd)

        self.att_weight = None
        self.decoder_only = decoder_only

        self.memory = memory.to(self.device)
        self.memory_encoded = None
        self.block_size = config.block_size
        self.apply(self._init_weights)
        

        logger.info(
            f"number of parameters: {sum(p.numel() for p in self.parameters())}"
        )
    
    def _memory(self, x):
        batch_size, seq_len, issue_num, embed_dim = x.size()
        position_embeddings = self.pos_emb_e[:, :seq_len, :, :]
        x = self.drop(x + position_embeddings)
        x = self.encoder_layers(x)
        x = self.ln_e(x)
        return x
    
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, issue_num, embed_dim = x.size()
        if self.decoder_only:
            memory = None
        else:
            if self.isLearning or self.isEncoded == False:
                memory = self.memory.repeat((batch_size,1,1,1))
                memory = self._memory(memory)
                self.isEncoded = True
                self.memory_encoded = memory
            else:
                memory = self.memory_encoded
        padding_mask = (torch.all(torch.eq(x,PAD_ID),dim=-1) == True).unsqueeze(-1).transpose(1,3).unsqueeze(-2)
        position_embeddings = self.pos_emb_d[:, :seq_len, :, :]
        x = self.drop(x + position_embeddings)
        for decoder in self.decoder_layers:
            x = decoder([x, memory, padding_mask])
        x = self.ln_d(x)
        outputs = self.head(x)

        return outputs
