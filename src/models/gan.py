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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class GANGenerator(nn.Module):
    def __init__(self, latent_dim, max_nodes, feature_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.max_nodes   = max_nodes
        self.feature_dim = feature_dim
        self.label_embed = nn.Embedding(num_classes, hidden_dim)
        self.input_proj  = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4), nn.LeakyReLU(0.2), nn.Dropout(0.3),
        )
        self.adj_gen  = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, max_nodes * max_nodes), nn.Sigmoid()
        )
        self.feat_gen = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, max_nodes * feature_dim), nn.Tanh()
        )

    def forward(self, z, labels):
        B = z.shape[0]
        x = torch.cat([z, self.label_embed(labels)], dim=1)
        h = self.input_proj(x)
        adj  = self.adj_gen(h).view(B, self.max_nodes, self.max_nodes)
        adj  = (adj + adj.transpose(1, 2)) / 2
        adj  = adj * (1 - torch.eye(self.max_nodes, device=adj.device).unsqueeze(0))
        feat = self.feat_gen(h).view(B, self.max_nodes, self.feature_dim)
        return adj, feat


class GANDiscriminator(nn.Module):
    def __init__(self, max_nodes, feature_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.label_embed    = nn.Embedding(num_classes, hidden_dim)
        self.adj_encoder    = nn.Sequential(
            nn.Linear(max_nodes, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2),
            nn.Dropout(0.3), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2)
        )
        self.feat_encoder   = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2),
            nn.Dropout(0.3), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2)
        )
        self.graph_proc     = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2)
        )
        self.output         = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Dropout(0.3), nn.Linear(hidden_dim, 1)
        )

    def forward(self, adj, feat, labels, mask):
        h  = torch.cat([self.adj_encoder(adj), self.feat_encoder(feat)], dim=-1)
        h  = h * mask.unsqueeze(-1)
        h  = self.graph_proc(h)
        ge = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.output(torch.cat([ge, self.label_embed(labels)], dim=1))


class GraphGAN:
    def __init__(self, latent_dim, max_nodes, feature_dim, num_classes, hidden_dim=256, device="cpu"):
        self.device      = device
        self.latent_dim  = latent_dim
        self.max_nodes   = max_nodes
        self.feature_dim = feature_dim
        self.generator    = GANGenerator(latent_dim, max_nodes, feature_dim, num_classes, hidden_dim).to(device)
        self.discriminator = GANDiscriminator(max_nodes, feature_dim, num_classes, hidden_dim).to(device)
        self.loss_fn      = nn.BCEWithLogitsLoss()

    def train_step(self, real_adj, real_feat, labels, mask, opt_G, opt_D):
        B          = real_adj.shape[0]
        real_lbl   = torch.ones(B, 1, device=self.device)
        fake_lbl   = torch.zeros(B, 1, device=self.device)
        # D step
        opt_D.zero_grad()
        z           = torch.randn(B, self.latent_dim, device=self.device)
        fake_adj, fake_feat = self.generator(z, labels)
        fake_adj   = fake_adj  * mask.unsqueeze(-1) * mask.unsqueeze(1)
        fake_feat  = fake_feat * mask.unsqueeze(-1)
        d_loss      = (
            self.loss_fn(self.discriminator(real_adj, real_feat, labels, mask), real_lbl)
          + self.loss_fn(self.discriminator(fake_adj.detach(), fake_feat.detach(), labels, mask), fake_lbl)
        ) / 2
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        opt_D.step()
        # G step
        opt_G.zero_grad()
        z2          = torch.randn(B, self.latent_dim, device=self.device)
        gen_adj, gen_feat = self.generator(z2, labels)
        gen_adj    = gen_adj  * mask.unsqueeze(-1) * mask.unsqueeze(1)
        gen_feat   = gen_feat * mask.unsqueeze(-1)
        g_loss      = self.loss_fn(self.discriminator(gen_adj, gen_feat, labels, mask), real_lbl)
        g_loss     += 0.1 * (gen_adj * (mask.unsqueeze(-1) * mask.unsqueeze(1))).mean()
        g_loss     += 0.05 * (gen_feat * mask.unsqueeze(-1)).pow(2).mean()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        opt_G.step()
        return d_loss.item(), g_loss.item()

    @torch.no_grad()
    def sample(self, label, num_nodes=None):
        self.generator.eval()
        if num_nodes is None:
            num_nodes = np.random.randint(15, min(25, self.max_nodes))
        num_nodes = min(num_nodes, self.max_nodes)
        z  = torch.randn(1, self.latent_dim, device=self.device)
        lt = torch.tensor([label], dtype=torch.long, device=self.device)
        adj, feat = self.generator(z, lt)
        af = adj[0, :num_nodes, :num_nodes]
        af = (af + af.T) / 2
        af = (af > 0.6).float()
        af.fill_diagonal_(0)
        if af.sum() < num_nodes - 1:
            for i in range(num_nodes - 1):
                if af[i:].sum() == 0:
                    j = (i + 1) % num_nodes
                    af[i, j] = af[j, i] = 1
        self.generator.train()
        return af.cpu(), feat[0, :num_nodes].cpu()