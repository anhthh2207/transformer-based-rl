import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        assert h_dim % n_heads == 0, 'hidden dimension must be divisible by number of heads'

        self.n_heads = n_heads
        self.max_T = max_T # vocab_size

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T) # mask for masked attention

        # register buffer makes sure mask does not get updated during back-propagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, state + action vector dimension

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = h_dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2) # (B, T, N, D) -> (B, N, T, D)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, 
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### projection heads (project to embedding)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = nn.Linear(1*context_len, h_dim*context_len) # reward-to-go
        self.embed_state = nn.Linear(state_dim*context_len, h_dim*context_len)

        # discrete actions
        self.embed_action = nn.Embedding(act_dim, h_dim)
        use_action_tanh = False # False for discrete actions

        self.embed_ln = nn.LayerNorm(h_dim)

        ### transformer blocks
        input_seq_len = 3 * context_len # 3 * context_len because we use reward, state and action as input, each is a vector of size h_dim
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

    def forward(self, timesteps, states, actions, returns_to_go):

        B = states.shape[0] # batch size, seq length * state_dim
        T = timesteps.shape[1] # seq length

        time_embeddings = self.embed_timestep(timesteps).reshape(B, T, self.h_dim)
        # print(states.shape, actions.shape, returns_to_go.shape, timesteps.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states).reshape(B, T, self.h_dim) + time_embeddings
        action_embeddings = self.embed_action(actions).reshape(B, T, self.h_dim) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go).reshape(B, T, self.h_dim) + time_embeddings
        # print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape)
        # stack rtg, states and actions and reshape sequence as
        # (r1, s1, a1, r2, s2, a2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).reshape(B, 3 * T, self.h_dim) # (B, 3 * T, h_dim)

        h = self.embed_ln(h)
        
        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1])  # predict action given r, s
    
        return state_preds, action_preds, return_preds
