import torch
import math
import numpy as np
from torch.nn import Module
import torch.nn.functional as F


class Attention_channels(Module):
    """ 
    Feature-wise attention across channels (timepoints), 
    where each ctDNA feature is a token represented by its values across the different timepoints. 
    """
    def __init__(self,
                 d_channel: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str):
        super(Attention_channels, self).__init__()
        
        self.device = device
        self._h = h
        self._q = q
        self.score = None
        self.sigmoid = torch.nn.Sigmoid()
        # Linear layers to project input features into query, key, and value vectors 
        # for multi-head attention
        self.W_q = torch.nn.Linear(d_channel, q * h)
        self.W_k = torch.nn.Linear(d_channel, q * h)
        self.W_v = torch.nn.Linear(d_channel, v * h)
        # Linear layer to project concatenated attention head outputs back to hidden dimension
        self.W_o = torch.nn.Linear(v * h, d_hidden)

    def forward(self, x):
        # Project input x into query, key, and value vectors for each attention head,
        # and concatenate the heads along the batch dimension for efficient computation
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)
        # Compute scaled dot-product attention scores for each head, and store in self.score 
        # for explainability analyses
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score
        score = score.to(self.device)
        # Apply softmax to attention scores to get attention weights
        score = F.softmax(score, dim=-1)
        # Compute attention output as weighted sum of value vectors, using the attention weights
        attention = torch.matmul(score, V)
        # Concatenate the outputs of the attention heads 
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        # Project the concatenated attention head outputs back to the hidden dimension using W_o
        self_attention = self.W_o(attention_heads)
        return self_attention, self.score    
     

class Attention_timesteps(Module):
    """ 
    Temporal attention across features, where each timestep is a token 
    encoding the value of all available ctDNA features at the given timestep.
    Each timestep corresponds to the collection time of a ctDNA sample (e.g. BL, C2D1, C3D1, C4D1).
    """

    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str):
        super(Attention_timesteps, self).__init__()

        self.device = device
        self._h = h
        self._q = q
        self.score = None
        self.sigmoid = torch.nn.Sigmoid()
        # Linear layers to project input features into query, key, and value vectors
        self.W_q = torch.nn.Linear(d_input, q * h)
        self.W_k = torch.nn.Linear(d_input, q * h)
        self.W_v = torch.nn.Linear(d_input, v * h)
        # Linear layer to project concatenated attention head outputs back to hidden dimension
        self.W_o = torch.nn.Linear(v * h, d_hidden)
 
    def forward(self, x):
        # Project input x into query, key, and value vectors for each attention head,
        # and concatenate the heads along the batch dimension for efficient computation
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)
        # Compute scaled dot-product attention scores for each head, and store in self.score
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score
        score = score.to(self.device)
        # Apply softmax to attention scores to get attention weights
        score = F.softmax(score, dim=-1)
        # Compute attention output as weighted sum of value vectors, using the attention weights
        attention = torch.matmul(score, V)
        # Concatenate the outputs of the attention heads 
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        # Project the concatenated attention head outputs back to the hidden dimension using W_o
        self_attention = self.W_o(attention_heads)
        return self_attention, self.score    
     
 

