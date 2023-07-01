import math
import logging
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    def __init__(self, state_dim, act_dim, context_len=30, n_blocks=6, embed_dim=128, n_heads=8, dropout_p=0.1):
        self.state_dim = state_dim          # state dim
        self.act_dim = act_dim              # action dim
        self.context_len = context_len      # context length
        self.n_blocks = n_blocks            # num of transformer blocks
        self.embed_dim = embed_dim          # embedding (hidden) dim of transformer
        self.n_heads = n_heads              # num of transformer heads
        self.dropout_p = dropout_p          # dropout probability

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        assert h_dim % n_heads == 0, 'hidden dimension must be divisible by number of heads'

        self.n_heads = n_heads
        self.h_dim = h_dim
        self.max_T = max_T # vocab_size

        self.c_attn = nn.Linear(h_dim, 3 * h_dim)

        self.c_proj = nn.Linear(h_dim, h_dim)

        self.attn_dropout = nn.Dropout(drop_p)
        self.resid_dropout = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T) # mask for masked attention

        # register buffer makes sure mask does not get updated during back-propagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, state + action vector dimension

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = h_dim

        # rearrange q, k, v as (B, N, T, D)
        q, k ,v  = self.c_attn(x).split(self.h_dim, dim=2)
        q = q.view(B, T, N, D).transpose(1,2) # (B, T, N, D) -> (B, N, T, D)
        k = k.view(B, T, N, D).transpose(1,2)
        v = v.view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        att = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to att
        att = att.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize att, all -inf -> 0 after softmax
        att = F.softmax(att, dim=-1)

        # attention (B, N, T, D)
        y = self.attn_dropout(att) @ v

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = y.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.resid_dropout(self.c_proj(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attn = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(h_dim, 4*h_dim),
            c_proj  = nn.Linear(4*h_dim, h_dim),
            act     = nn.GELU(),
            dropout = nn.Dropout(drop_p),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward
        self.ln_1 = nn.LayerNorm(h_dim)
        self.ln_2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, 
                 n_heads, drop_p, max_timestep=10000):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len

        ### projection heads (project to embedding)
        self.embed_timestep = nn.Sequential(nn.Embedding(max_timestep, h_dim), nn.Tanh())

        self.embed_rtg = nn.Sequential(nn.Linear(1, h_dim), nn.Tanh())
        
        # self.embed_state = nn.Linear(state_dim*context_len, h_dim*context_len)
        self.embed_state = nn.Sequential(
                            nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                            nn.Flatten(), nn.Linear(3136, h_dim), nn.Tanh())
        
        use_action_tanh = False # False for discrete actions

        self.embed_ln = nn.LayerNorm(h_dim)

        ### transformer blocks
        input_seq_len = 3 * context_len # 3 * context_len because we use reward, state and action as input, each is a vector of size h_dim
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]),
            ln_f = nn.LayerNorm(h_dim),
        ))

        ### prediction heads
        self.predict_action = nn.Linear(h_dim, act_dim, bias=False)

        # init all weights
        self.apply(self._init_weights)

        self.embed_action = nn.Sequential(nn.Embedding(act_dim, h_dim), nn.Tanh())
        nn.init.normal_(self.embed_action[0].weight, mean=0.0, std=0.02)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, timesteps, states, actions, targets, returns_to_go):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size)
        # targets: (batch, block_size)
        # rtgs: (batch, block_size)
        # timesteps: (batch, block_size)

        B = states.shape[0] # batch size, context length, state_dim, state_dim
        T = states.shape[1] # context length

        time_embeddings = self.embed_timestep(timesteps.type(torch.long)).reshape(B, T, self.h_dim)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states.reshape(B*T,4,84,84).type(torch.float32))
        state_embeddings = state_embeddings.reshape(B, T, self.h_dim) + time_embeddings
        action_embeddings = self.embed_action(actions.type(torch.long)).reshape(B, T - int(targets is None), self.h_dim) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go.reshape(B,T,1).type(torch.float32)).reshape(B, T, self.h_dim) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r1, s1, a1, r2, s2, a2 ...)
        x = torch.zeros(B, 3 * T - int(targets is None), self.h_dim, device=states.device)
        x[:,0::3,:] = returns_embeddings
        x[:,1::3,:] = state_embeddings
        x[:,2::3,:] = action_embeddings

        x = self.embed_ln(x)
        
        # transformer and prediction
        for block in self.transformer.h:
            x = block(x)
        h = self.transformer.ln_f(x)

        # get predictions
        action_preds = self.predict_action(h)
        logits = action_preds[:, 1::3, :] # take only prediction based on states and returns
    
        # In the original paper, it is stated that predicting the states and returns are not necessary
        # and does not improve the performance. However, it could be an interesting study for future work.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.act_dim), targets.reshape(-1))
        return logits, loss