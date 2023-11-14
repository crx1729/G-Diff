import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class GDN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, mlp_dims, emb_size, graph, graph_layers, norm=False, dropout=0.5):
        super(GDN, self).__init__()
        self.mlp_dims = mlp_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        self.graph = graph
        self.graph_layers = graph_layers
        self.mlp_dims[0] += self.time_emb_dim
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        self.mlp = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(self.mlp_dims[:-1], self.mlp_dims[1:])
        ])
        for l in self.mlp:
            nn.init.xavier_normal_(l.weight)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def graph_layer(self, x):
        for i in range(self.graph_layers):
            x = torch.sparse.mm(self.graph, x.t()).t() + x
            if self.norm:
                x = F.normalize(x)
        return x

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        xx = self.graph_layer(x) + x
        if self.norm:
            xx = F.normalize(xx)
        h = torch.cat([self.drop(xx), emb], dim=1)
        for l in self.mlp:
            h = self.drop(self.act(l(h)))
        return h + xx


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
