import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool,GINConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import os
import random
import matplotlib.pyplot as plt
import networkx as nx
import glob
import gc

#from model import GINClassification
#from utils import draw_random_graph_samples, train_model, extract_embeddings, collect_labels, predict_labels, visualize_embeddings

from torch_geometric.utils import to_networkx
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, Airports


from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
#from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from scipy import linalg
from typing import List, Tuple, Optional
from grakel import Graph
import copy
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from scipy.stats import wasserstein_distance

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

from grakel import Graph as GrakelGraph
from torch_geometric.datasets import PolBlogs

from torch_geometric.utils import k_hop_subgraph
from collections import Counter

import config

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")




class LaplaceDiffusion:
    def __init__(self, num_timesteps=100, epsilon=0.5):
        self.num_timesteps = num_timesteps
        self.epsilon       = epsilon
        self.betas         = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas             = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod          = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod    = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod  = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def _to(self, device):
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))

    def sample_laplace_noise(self, shape, device):
        u = torch.rand(shape, device=device) - 0.5
        b = self.epsilon / np.sqrt(2)
        noise = -b * torch.sign(u) * torch.log(torch.clamp(1 - 2 * torch.abs(u), min=1e-10))
        return torch.clamp(noise, -5, 5)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.sample_laplace_noise(x_start.shape, x_start.device)
        sa  = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        soa = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sa * x_start + soa * noise, noise

    def predict_start_from_noise(self, x_t, t, noise):
        sr  = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1)
        srm = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)
        return sr * x_t - srm * noise

    def q_posterior(self, x_start, x_t, t):
        c1 = self.posterior_mean_coef1[t].view(-1, 1, 1)
        c2 = self.posterior_mean_coef2[t].view(-1, 1, 1)
        return c1 * x_start + c2 * x_t


class GraphDiffusionNetwork(nn.Module):
    def __init__(self, max_nodes, feature_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.max_nodes   = max_nodes
        self.feature_dim = feature_dim
        self.hidden_dim  = hidden_dim
        time_dim         = 128
        self.time_dim    = time_dim
        self.time_embed  = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.label_embed = nn.Embedding(num_classes, hidden_dim)
        self.adj_encoder = nn.Sequential(
            nn.Linear(max_nodes,     hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),    nn.LayerNorm(hidden_dim),       nn.SiLU(),
        )
        self.feat_encoder = nn.Sequential(
            nn.Linear(feature_dim,   hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),    nn.LayerNorm(hidden_dim),       nn.SiLU(),
        )
        self.joint_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2), nn.SiLU(), nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.SiLU()
            ),
        ])
        self.adj_decoder  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, max_nodes)
        )
        self.feat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, feature_dim)
        )

    def get_time_embedding(self, timesteps):
        half = self.time_dim // 2
        emb  = np.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
        emb  = timesteps[:, None] * emb[None, :]
        emb  = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb) * 0.1

    def forward(self, adj_t, feat_t, t, label, mask):
        B, N   = adj_t.shape[:2]
        t_emb  = self.get_time_embedding(t.float())
        l_emb  = self.label_embed(label)
        cond   = (t_emb + l_emb).unsqueeze(1).expand(-1, N, -1)
        h_adj  = self.adj_encoder(adj_t)
        h_feat = self.feat_encoder(feat_t)
        h      = torch.cat([h_adj, h_feat, cond], dim=-1)
        for layer in self.joint_processor:
            h = layer(h)
        h = h * mask.unsqueeze(-1)
        return self.adj_decoder(h), self.feat_decoder(h)


class GraphDiffusionModel(nn.Module):
    def __init__(self, max_nodes, feature_dim, num_classes,
                 num_timesteps=100, epsilon=0.5, hidden_dim=256, device="cpu"):
        super().__init__()
        self.device      = device
        self.max_nodes   = max_nodes
        self.feature_dim = feature_dim
        self.diffusion   = LaplaceDiffusion(num_timesteps, epsilon)
        self.model       = GraphDiffusionNetwork(
            max_nodes, feature_dim, num_classes, hidden_dim
        ).to(device)
        self.diffusion._to(device)

    def train_step(self, adj, feat, labels, mask, optimizer):
        self.model.train()
        B = adj.shape[0]
        adj_norm = torch.clamp(adj, 0, 1) * 2.0 - 1.0
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
        adj_t,  adj_noise  = self.diffusion.q_sample(adj_norm, t)
        feat_t, feat_noise = self.diffusion.q_sample(feat,     t)
        adj_pred, feat_pred = self.model(adj_t, feat_t, t, labels, mask)
        mask2d = mask.unsqueeze(-1) * mask.unsqueeze(1)
        mask1d = mask.unsqueeze(-1)
        adj_loss  = ((adj_pred  - adj_noise)  * mask2d).pow(2).sum() / (mask2d.sum() + 1e-8)
        feat_loss = ((feat_pred - feat_noise) * mask1d).pow(2).sum() / (mask1d.sum() + 1e-8)
        loss = adj_loss + 2.0 * feat_loss
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0, 0.0, 0.0
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        optimizer.step()
        return loss.item(), adj_loss.item(), feat_loss.item()

    @torch.no_grad()
    def sample(self, label, num_nodes=None):
        self.model.eval()
        if num_nodes is None:
            num_nodes = np.random.randint(15, min(25, self.max_nodes))
        num_nodes = min(num_nodes, self.max_nodes)
        adj_t  = self.diffusion.sample_laplace_noise(
            (1, self.max_nodes, self.max_nodes), self.device) * 0.5
        feat_t = self.diffusion.sample_laplace_noise(
            (1, self.max_nodes, self.feature_dim), self.device) * 0.5
        mask   = torch.zeros(1, self.max_nodes, device=self.device)
        mask[0, :num_nodes] = 1
        label_t = torch.tensor([label], dtype=torch.long, device=self.device)
        for ti in reversed(range(self.diffusion.num_timesteps)):
            t = torch.tensor([ti], device=self.device)
            ap, fp = self.model(adj_t, feat_t, t, label_t, mask)
            a0 = torch.clamp(self.diffusion.predict_start_from_noise(adj_t,  t, ap), -1, 1)
            f0 = torch.clamp(self.diffusion.predict_start_from_noise(feat_t, t, fp), -3, 3)
            if ti > 0:
                adj_t  = self.diffusion.q_posterior(a0, adj_t,  t)
                feat_t = self.diffusion.q_posterior(f0, feat_t, t)
                var    = self.diffusion.posterior_variance[t].view(-1, 1, 1)
                ns     = torch.sqrt(var) * 0.5
                adj_t  += ns * self.diffusion.sample_laplace_noise(adj_t.shape,  self.device)
                feat_t += ns * self.diffusion.sample_laplace_noise(feat_t.shape, self.device)
            else:
                adj_t, feat_t = a0, f0
        af = (adj_t[0, :num_nodes, :num_nodes] + 1.0) / 2.0
        af = (af + af.T) / 2
        th = torch.quantile(af.flatten(), 0.7)
        af = (af > th).float()
        af.fill_diagonal_(0)
        return af.cpu(), feat_t[0, :num_nodes].cpu()