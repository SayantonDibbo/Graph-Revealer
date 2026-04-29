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

class CVAEEncoder(nn.Module):
    def __init__(self, max_nodes, feature_dim, num_classes, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.label_embed  = nn.Embedding(num_classes, hidden_dim)
        self.adj_encoder  = nn.Sequential(
            nn.Linear(max_nodes, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.feat_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.joint        = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.pool         = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_mu        = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar    = nn.Linear(hidden_dim, latent_dim)

    def forward(self, adj, feat, label, mask):
        B, N  = adj.shape[:2]
        le    = self.label_embed(label)
        h     = torch.cat([self.adj_encoder(adj), self.feat_encoder(feat)], dim=-1)
        h     = self.joint(h) * mask.unsqueeze(-1)
        hg    = h.sum(1) / mask.sum(1, keepdim=True)
        hg    = self.pool(hg + le)
        return self.fc_mu(hg), self.fc_logvar(hg)


class CVAEDecoder(nn.Module):
    def __init__(self, max_nodes, feature_dim, num_classes, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.max_nodes   = max_nodes
        self.feature_dim = feature_dim
        self.label_embed  = nn.Embedding(num_classes, hidden_dim)
        self.latent_proj  = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.node_expand  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * max_nodes),
            nn.LayerNorm(hidden_dim * max_nodes), nn.ReLU()
        )
        self.adj_dec  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, max_nodes)
        )
        self.feat_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, z, label, mask):
        B  = z.shape[0]; N = self.max_nodes
        h  = (self.latent_proj(z) + self.label_embed(label))
        h  = self.node_expand(h).view(B, N, -1) * mask.unsqueeze(-1)
        al = self.adj_dec(h)
        al = (al + al.transpose(1, 2)) / 2
        return al, self.feat_dec(h)


class GraphCVAE(nn.Module):
    def __init__(self, max_nodes, feature_dim, num_classes,
                 hidden_dim=256, latent_dim=128, device="cpu"):
        super().__init__()
        self.device      = device
        self.max_nodes   = max_nodes
        self.latent_dim  = latent_dim
        self.encoder     = CVAEEncoder(max_nodes, feature_dim, num_classes, hidden_dim, latent_dim).to(device)
        self.decoder     = CVAEDecoder(max_nodes, feature_dim, num_classes, hidden_dim, latent_dim).to(device)

    @staticmethod
    def _laplace(shape, device, eps=0.5):
        u = torch.rand(shape, device=device) - 0.5
        b = eps / np.sqrt(2)
        return torch.clamp(-b * torch.sign(u) * torch.log(
            torch.clamp(1 - 2 * torch.abs(u), min=1e-10)), -10, 10)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * self._laplace(mu.shape, mu.device)

    def forward(self, adj, feat, label, mask):
        mu, lv = self.encoder(adj, feat, label, mask)
        z      = self.reparameterize(mu, lv)
        al, fr = self.decoder(z, label, mask)
        return al, fr, mu, lv

    def loss(self, adj, feat, al, fr, mu, lv, mask, beta=1.0):
        B   = adj.shape[0]
        m2d = mask.unsqueeze(-1) * mask.unsqueeze(1)
        m1d = mask.unsqueeze(-1)
        rl_adj  = F.binary_cross_entropy_with_logits(al * m2d, adj * m2d, reduction="sum") / m2d.sum()
        rl_feat = F.mse_loss(fr * m1d, feat * m1d, reduction="sum") / m1d.sum()
        kl      = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / B
        sp      = (torch.sigmoid(al) * m2d).sum() / m2d.sum()
        return rl_adj + rl_feat + beta * kl + 0.05 * sp, rl_adj, rl_feat, kl, sp

    def train_step(self, adj, feat, labels, mask, optimizer, beta=1.0):
        self.train()
        al, fr, mu, lv = self(adj, feat, labels, mask)
        tl, *_ = self.loss(adj, feat, al, fr, mu, lv, mask, beta)
        if torch.isnan(tl) or torch.isinf(tl):
            return tuple([0.0] * 5)
        optimizer.zero_grad(); tl.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return tl.item(), *[x.item() for x in _]

    @torch.no_grad()
    def sample(self, label, num_nodes=None):
        self.eval()
        if num_nodes is None:
            num_nodes = np.random.randint(15, min(25, self.max_nodes))
        num_nodes = min(num_nodes, self.max_nodes)
        z    = self._laplace((1, self.latent_dim), self.device)
        mask = torch.zeros(1, self.max_nodes, device=self.device)
        mask[0, :num_nodes] = 1
        lt   = torch.tensor([label], dtype=torch.long, device=self.device)
        al, fr = self.decoder(z, lt, mask)
        ap   = torch.sigmoid(al[0, :num_nodes, :num_nodes])
        ap   = (ap + ap.T) / 2
        af   = (ap > 0.5).float(); af.fill_diagonal_(0)
        return af.cpu(), fr[0, :num_nodes].cpu()