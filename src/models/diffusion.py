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
    def __init__(self, num_timesteps, epsilon):
        self.num_timesteps = num_timesteps
        self.epsilon       = epsilon

        self.betas                       = torch.linspace(config.BETA_START, config.BETA_END, num_timesteps)
        self.alphas                      = 1.0 - self.betas
        self.alphas_cumprod              = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev         = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod         = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod   = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance          = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    def sample_laplace_noise(self, shape, device):
        u = torch.rand(shape, device=device) - 0.5
        b = self.epsilon / np.sqrt(2)
        noise = -b * torch.sign(u) * torch.log(torch.clamp(1 - 2 * torch.abs(u), min=1e-10))
        return torch.clamp(noise, config.LAPLACE_CLAMP_MIN, config.LAPLACE_CLAMP_MAX)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.sample_laplace_noise(x_start.shape, x_start.device)
        sqrt_a   = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_a * x_start + sqrt_1ma * noise, noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t
                - self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * noise)

    def q_posterior(self, x_start, x_t, t):
        return (self.posterior_mean_coef1[t].view(-1, 1, 1) * x_start
                + self.posterior_mean_coef2[t].view(-1, 1, 1) * x_t)


class GraphDiffusionNetwork(nn.Module):
    def __init__(self, max_nodes, feature_dim, num_classes,
                 hidden_dim=config.GDN_HIDDEN_DIM):
        super().__init__()
        self.max_nodes   = max_nodes
        self.feature_dim = feature_dim
        self.hidden_dim  = hidden_dim
        time_dim         = config.GDN_TIME_DIM
        self.time_dim    = time_dim

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.label_embed = nn.Embedding(num_classes, hidden_dim)

        self.adj_encoder = nn.Sequential(
            nn.Linear(max_nodes, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.feat_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        self.joint_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2), nn.SiLU(),
                nn.Dropout(config.GDN_DROPOUT)),
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.SiLU(),
                nn.Dropout(config.GDN_DROPOUT)),
        ])
        self.adj_decoder  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes))
        self.feat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim))

    def get_time_embedding(self, timesteps):
        half = self.time_dim // 2
        emb  = np.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
        emb  = timesteps[:, None] * emb[None, :]
        emb  = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb) * config.GDN_TIME_SCALE

    def forward(self, adj_t, feat_t, t, label, mask):
        B, N   = adj_t.shape[:2]
        t_emb  = self.get_time_embedding(t.float())
        l_emb  = self.label_embed(label)
        cond   = (t_emb + l_emb).unsqueeze(1).expand(-1, N, -1)
        h      = torch.cat([self.adj_encoder(adj_t), self.feat_encoder(feat_t), cond], dim=-1)
        for layer in self.joint_processor:
            h = layer(h)
        h = h * mask.unsqueeze(-1)
        return self.adj_decoder(h), self.feat_decoder(h)


class GraphDiffusionModel(nn.Module):
    def __init__(self, max_nodes, feature_dim, num_classes,
                 num_timesteps, epsilon, device):
        super().__init__()
        self.device      = device
        self.max_nodes   = max_nodes
        self.feature_dim = feature_dim
        self.diffusion   = LaplaceDiffusion(num_timesteps, epsilon)
        self.model       = GraphDiffusionNetwork(max_nodes, feature_dim, num_classes).to(device)

        for attr in dir(self.diffusion):
            val = getattr(self.diffusion, attr)
            if isinstance(val, torch.Tensor):
                setattr(self.diffusion, attr, val.to(device))

    def train_step(self, adj, features, labels, mask, optimizer):
        self.model.train()
        B        = adj.shape[0]
        adj_norm = torch.clamp(adj, 0, 1) * 2.0 - 1.0
        t        = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)

        adj_t,  adj_noise_true  = self.diffusion.q_sample(adj_norm,  t)
        feat_t, feat_noise_true = self.diffusion.q_sample(features, t)
        adj_noise_pred, feat_noise_pred = self.model(adj_t, feat_t, t, labels, mask)

        mask_2d  = mask.unsqueeze(-1) * mask.unsqueeze(1)
        mask_1d  = mask.unsqueeze(-1)
        adj_loss  = ((adj_noise_pred  - adj_noise_true)  * mask_2d) ** 2
        adj_loss  = adj_loss.sum() / (mask_2d.sum() + 1e-8)
        feat_loss = ((feat_noise_pred - feat_noise_true) * mask_1d) ** 2
        feat_loss = feat_loss.sum() / (mask_1d.sum() + 1e-8)
        loss      = adj_loss + config.FEAT_LOSS_WEIGHT * feat_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠  NaN/Inf detected — skipping batch")
            return 0.0, 0.0, 0.0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.DIFF_GRAD_CLIP)
        optimizer.step()
        return loss.item(), adj_loss.item(), feat_loss.item()

    @torch.no_grad()
    def sample(self, label, num_nodes=None):
        self.model.eval()
        if num_nodes is None:
            num_nodes = np.random.randint(
                config.SAMPLE_MIN_NODES,
                min(config.SAMPLE_MAX_NODES, self.max_nodes))
        num_nodes = min(num_nodes, self.max_nodes)

        adj_t  = self.diffusion.sample_laplace_noise(
            (1, self.max_nodes, self.max_nodes), self.device) * config.INITIAL_NOISE_SCALE
        feat_t = self.diffusion.sample_laplace_noise(
            (1, self.max_nodes, self.feature_dim), self.device) * config.INITIAL_NOISE_SCALE

        mask          = torch.zeros(1, self.max_nodes, device=self.device)
        mask[0, :num_nodes] = 1
        label_tensor  = torch.tensor([label], dtype=torch.long, device=self.device)

        for t_idx in reversed(range(self.diffusion.num_timesteps)):
            t = torch.tensor([t_idx], device=self.device)
            adj_np, feat_np = self.model(adj_t, feat_t, t, label_tensor, mask)

            adj_0  = torch.clamp(
                self.diffusion.predict_start_from_noise(adj_t,  t, adj_np),
                config.ADJ_CLAMP_MIN, config.ADJ_CLAMP_MAX)
            feat_0 = torch.clamp(
                self.diffusion.predict_start_from_noise(feat_t, t, feat_np),
                config.FEAT_CLAMP_MIN, config.FEAT_CLAMP_MAX)

            if t_idx > 0:
                adj_t  = self.diffusion.q_posterior(adj_0,  adj_t,  t)
                feat_t = self.diffusion.q_posterior(feat_0, feat_t, t)
                var    = self.diffusion.posterior_variance[t].view(-1, 1, 1)
                scale  = torch.sqrt(var) * config.POSTERIOR_NOISE_SCALE
                adj_t  += scale * self.diffusion.sample_laplace_noise(adj_t.shape,  self.device)
                feat_t += scale * self.diffusion.sample_laplace_noise(feat_t.shape, self.device)
            else:
                adj_t, feat_t = adj_0, feat_0

        adj_final  = (adj_t[0, :num_nodes, :num_nodes] + 1.0) / 2.0
        adj_final  = (adj_final + adj_final.T) / 2
        threshold  = torch.quantile(adj_final.flatten(), config.ADJ_THRESHOLD_PERCENTILE)
        adj_final  = (adj_final > threshold).float()
        adj_final.fill_diagonal_(0)

        if adj_final.sum() < num_nodes - 1:
            adj_probs = (adj_t[0, :num_nodes, :num_nodes] + 1.0) / 2.0
            adj_probs = (adj_probs + adj_probs.T) / 2
            for i in range(num_nodes - 1):
                if adj_final[i].sum() == 0:
                    best_j = (i + 1) % num_nodes
                    for j in range(num_nodes):
                        if i != j and adj_probs[i, j] > adj_probs[i, best_j]:
                            best_j = j
                    adj_final[i, best_j] = adj_final[best_j, i] = 1

        return adj_final.cpu(), feat_t[0, :num_nodes].cpu()


def train_diffusion(model, dataloader, num_epochs, lr):
    optimizer   = torch.optim.AdamW(
        model.model.parameters(), lr=lr,
        weight_decay=config.DIFF_WEIGHT_DECAY,
        betas=(config.DIFF_ADAM_BETA1, config.DIFF_ADAM_BETA2))
    warmup      = config.DIFF_WARMUP_EPOCHS

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(num_epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"\nTraining diffusion model for {num_epochs} epochs …")

    for epoch in range(num_epochs):
        total_loss = total_adj = total_feat = count = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for adj, feat, labels, mask in pbar:
            adj    = adj.to(model.device)
            feat   = feat.to(model.device)
            labels = labels.to(model.device)
            mask   = mask.to(model.device)
            loss, al, fl = model.train_step(adj, feat, labels, mask, optimizer)
            if loss > 0:
                total_loss += loss; total_adj += al; total_feat += fl; count += 1
                pbar.set_postfix({"loss": f"{loss:.4f}"})
        scheduler.step()
        if count:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss={total_loss/count:.6f} "
                  f"(Adj={total_adj/count:.6f}, Feat={total_feat/count:.6f}) | "
                  f"LR={scheduler.get_last_lr()[0]:.6f}")

    print("✓ Diffusion training complete")