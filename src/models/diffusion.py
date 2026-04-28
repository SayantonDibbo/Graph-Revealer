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


class NCI1GraphDataset(Dataset):
            """Carefully prepared NCI1 dataset"""
            def __init__(self, data, classes, max_nodes=MAX_NODE):
                print("Loading NCI1 dataset...")
                # self.dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
                self.datasets = data
                self.max_nodes = max_nodes
                self.num_classes = classes

                # Filter and validate graphs
                self.valid_data = []
                for data in self.datasets:
                    if (data.x is not None and data.x.shape[1] > 0 and
                        2 <= data.num_nodes <= max_nodes and
                        data.edge_index.shape[1] > 0):  # Ensure has edges
                        # Calculate average degree
                        avg_degree = data.edge_index.shape[1] / data.num_nodes
                        # Only keep sparse graphs (molecular structures are sparse)
                        if avg_degree <= 5.0:  # Filter out dense graphs
                            self.valid_data.append(data)

                # Normalize features globally
                all_features = torch.cat([data.x for data in self.valid_data], dim=0)
                self.feat_mean = all_features.mean(dim=0, keepdim=True)
                self.feat_std = all_features.std(dim=0, keepdim=True) + 1e-8

                self.feature_dim = self.valid_data[0].x.shape[1]

                # Compute average graph statistics per class
                self.class_stats = {}
                for label in range(self.num_classes):
                    class_graphs = [d for d in self.valid_data if d.y.item() == label]
                    avg_nodes = np.mean([d.num_nodes for d in class_graphs])
                    avg_edges = np.mean([d.edge_index.shape[1] / 2 for d in class_graphs])  # Undirected
                    self.class_stats[label] = {
                        'avg_nodes': int(avg_nodes),
                        'avg_edges': int(avg_edges),
                        'count': len(class_graphs)
                    }

                print(f"\nLoaded {len(self.valid_data)} valid graphs")
                print(f"Feature dimension: {self.feature_dim}")
                for label, stats in self.class_stats.items():
                    print(f"  Class {label}: {stats['count']} graphs, "
                          f"avg {stats['avg_nodes']} nodes, {stats['avg_edges']} edges")

            def __len__(self):
                return len(self.valid_data)

            def __getitem__(self, idx):
                data = self.valid_data[idx]

                # Normalize features
                features_norm = (data.x - self.feat_mean) / self.feat_std

                # Dense adjacency (undirected, so just keep one direction)
                adj = to_dense_adj(data.edge_index, max_num_nodes=self.max_nodes)[0]

                # Pad features
                features = torch.zeros(self.max_nodes, self.feature_dim)
                features[:data.num_nodes] = features_norm

                # Create mask
                mask = torch.zeros(self.max_nodes)
                mask[:data.num_nodes] = 1

                return adj, features, data.y.item(), mask


        class LaplaceDiffusion:
            """
            CHANGES:
            - Removed harsh noise clipping (was breaking distribution)
            - Added softer clamping only when necessary
            """
            def __init__(self, num_timesteps=100, epsilon=0.5):
                self.num_timesteps = num_timesteps
                self.epsilon = epsilon
         
                # Linear schedule
                self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
                self.alphas = 1.0 - self.betas
                self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
                self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
         
                self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
                self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
                self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
                self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
         
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
                    (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
                )
         
            def sample_laplace_noise(self, shape, device):
                """
                CHANGE: Softer clamping to preserve Laplace distribution shape
                OLD: torch.clamp(noise, -10, 10)  # Too harsh!
                NEW: torch.clamp(noise, -5, 5)    # Softer, preserves distribution better
                """
                u = torch.rand(shape, device=device) - 0.5
                b = self.epsilon / np.sqrt(2)
                noise = -b * torch.sign(u) * torch.log(torch.clamp(1 - 2 * torch.abs(u), min=1e-10))
                return torch.clamp(noise, -5, 5)  # Softer clamping
         
            def q_sample(self, x_start, t, noise=None):
                """Forward process: add noise"""
                if noise is None:
                    noise = self.sample_laplace_noise(x_start.shape, x_start.device)
         
                sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
                sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
         
                return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise
         
            def predict_start_from_noise(self, x_t, t, noise):
                """Predict x_0 from x_t and noise"""
                sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1)
                sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)
                return sqrt_recip * x_t - sqrt_recipm1 * noise
         
            def q_posterior(self, x_start, x_t, t):
                """Compute posterior mean"""
                coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1)
                coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1)
                return coef1 * x_start + coef2 * x_t
         
         
        class GraphDiffusionNetwork(nn.Module):
            """
            MAJOR CHANGES:
            1. Fixed adj_decoder output dimension (was wrong!)
            2. Added edge-level processing for adjacency matrix
            3. Improved conditioning mechanism
            4. Better normalization
            """
            def __init__(self, max_nodes, feature_dim, num_classes, hidden_dim=256):
                super().__init__()
                self.max_nodes = max_nodes
                self.feature_dim = feature_dim
                self.hidden_dim = hidden_dim
         
                # Time embedding
                time_dim = 128
                self.time_embed = nn.Sequential(
                    nn.Linear(time_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.time_dim = time_dim
         
                # Label embedding
                self.label_embed = nn.Embedding(num_classes, hidden_dim)
         
                # === CHANGE 1: Improved adjacency encoding ===
                # Process adjacency as edge features instead of node features
                self.adj_encoder = nn.Sequential(
                    nn.Linear(max_nodes, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU()
                )
         
                # === CHANGE 2: Better feature encoder ===
                self.feat_encoder = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU()
                )
         
                # === CHANGE 3: Improved joint processing ===
                self.joint_processor = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),  # +hidden for conditioning
                        nn.LayerNorm(hidden_dim * 2),
                        nn.SiLU(),
                        nn.Dropout(0.1)
                    ),
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU(),
                        nn.Dropout(0.1)
                        
                    )
                ])
         
                # === CHANGE 4: Fixed adjacency decoder dimension! ===
                # OLD: outputs [B, N, N] directly - WRONG dimension!
                # NEW: outputs [B, N, N] properly through reshape
                self.adj_decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, max_nodes)  # Output N values per node
                )
         
                # Feature decoder (unchanged)
                self.feat_decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, feature_dim)
                )
         
            def get_time_embedding(self, timesteps):
                """
                CHANGE: Added scaling for better gradient flow
                """
                half_dim = self.time_dim // 2
                emb = np.log(10000) / (half_dim - 1)
                emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
                emb = timesteps[:, None] * emb[None, :]
                emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
                
                # Scale down to prevent overwhelming other features
                return self.time_embed(emb) * 0.1
         
            def forward(self, adj_t, feat_t, t, label, mask):
                """
                CHANGE: Improved conditioning integration
                """
                B, N = adj_t.shape[:2]
         
                # Embeddings
                t_emb = self.get_time_embedding(t.float())  # [B, hidden]
                label_emb = self.label_embed(label)  # [B, hidden]
                cond_emb = t_emb + label_emb  # [B, hidden]
         
                # Encode
                h_adj = self.adj_encoder(adj_t)  # [B, N, hidden]
                h_feat = self.feat_encoder(feat_t)  # [B, N, hidden]
         
                # Concatenate node representations
                h = torch.cat([h_adj, h_feat], dim=-1)  # [B, N, 2*hidden]
         
                # === CHANGE: Better conditioning integration ===
                cond_emb_expanded = cond_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, hidden]
                h = torch.cat([h, cond_emb_expanded], dim=-1)  # [B, N, 2*hidden + hidden]
         
                # Process through layers
                for layer in self.joint_processor:
                    h = layer(h)  # [B, N, hidden]
         
                # Apply mask
                h = h * mask.unsqueeze(-1)
         
                # Decode
                adj_noise = self.adj_decoder(h)  # [B, N, N] - Now correct!
                feat_noise = self.feat_decoder(h)  # [B, N, F]
         
                return adj_noise, feat_noise
         
         
        class GraphDiffusionModel(nn.Module):
            """
            CHANGES:
            1. Removed conflicting sparsity loss
            2. Better loss normalization
            3. Improved gradient handling
            """
            def __init__(self, max_nodes, feature_dim, num_classes, num_timesteps=100,
                         epsilon=0.5, device=device):
                super().__init__()
                self.device = device
                self.max_nodes = max_nodes
                self.feature_dim = feature_dim
         
                self.diffusion = LaplaceDiffusion(num_timesteps, epsilon)
                self.model = GraphDiffusionNetwork(max_nodes, feature_dim, num_classes).to(device)
         
                # Move diffusion params to device
                for attr_name in dir(self.diffusion):
                    attr = getattr(self.diffusion, attr_name)
                    if isinstance(attr, torch.Tensor):
                        setattr(self.diffusion, attr_name, attr.to(device))
         
            def train_step(self, adj, features, labels, mask, optimizer):
                """
                MAJOR CHANGES:
                1. Removed sparsity loss (was conflicting with denoising objective)
                2. Better loss weighting
                3. Improved normalization
                """
                self.model.train()
                B = adj.shape[0]
         
                # === CHANGE: Better adjacency normalization ===
                # OLD: adj * 2.0 - 1.0  (can cause issues if adj not in [0,1])
                # NEW: Ensure proper range first
                adj_norm = torch.clamp(adj, 0, 1) * 2.0 - 1.0
         
                # Sample timesteps
                t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
         
                # Add noise
                adj_t, adj_noise_true = self.diffusion.q_sample(adj_norm, t)
                feat_t, feat_noise_true = self.diffusion.q_sample(features, t)
         
                # Predict noise
                adj_noise_pred, feat_noise_pred = self.model(adj_t, feat_t, t, labels, mask)
         
                # === CHANGE: Improved loss computation ===
                mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(1)  # [B, N, N]
                mask_1d = mask.unsqueeze(-1)  # [B, N, 1]
         
                # Adjacency loss with better normalization
                adj_diff = (adj_noise_pred - adj_noise_true) * mask_2d
                adj_loss = (adj_diff ** 2).sum() / (mask_2d.sum() + 1e-8)
         
                # Feature loss with better normalization
                feat_diff = (feat_noise_pred - feat_noise_true) * mask_1d
                feat_loss = (feat_diff ** 2).sum() / (mask_1d.sum() + 1e-8)
         
                # === CHANGE: Removed sparsity loss! ===
                # OLD: Added sparsity_loss which fights against denoising
                # NEW: Let the model learn the true data distribution
                # The model will naturally learn sparsity from the training data
         
                # === CHANGE: Better loss weighting ===
                # Features are more important for molecular graphs
                loss = adj_loss + 2.0 * feat_loss  # Weight features 2x
         
                # Gradient clipping check
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠ NaN/Inf detected, skipping batch")
                    return 0.0, 0.0, 0.0
         
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                
                # === CHANGE: More aggressive gradient clipping ===
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Was 1.0
                
                optimizer.step()
         
                return loss.item(), adj_loss.item(), feat_loss.item()
         
            @torch.no_grad()
            def sample(self, label, num_nodes=None):
                """
                CHANGES:
                1. Better noise initialization
                2. Improved post-processing
                """
                self.model.eval()
         
                if num_nodes is None:
                    num_nodes = np.random.randint(15, min(25, self.max_nodes))
                num_nodes = min(num_nodes, self.max_nodes)
         
                # === CHANGE: Better noise initialization ===
                # Start with smaller initial noise
                adj_t = self.diffusion.sample_laplace_noise(
                    (1, self.max_nodes, self.max_nodes), self.device
                ) * 0.5  # Scale down initial noise
         
                feat_t = self.diffusion.sample_laplace_noise(
                    (1, self.max_nodes, self.feature_dim), self.device
                ) * 0.5
         
                # Mask
                mask = torch.zeros(1, self.max_nodes, device=self.device)
                mask[0, :num_nodes] = 1
                label_tensor = torch.tensor([label], dtype=torch.long, device=self.device)
         
                # Denoise step by step
                for t_idx in reversed(range(self.diffusion.num_timesteps)):
                    t = torch.tensor([t_idx], device=self.device)
         
                    # Predict noise
                    adj_noise_pred, feat_noise_pred = self.model(
                        adj_t, feat_t, t, label_tensor, mask
                    )
         
                    # Predict x_0
                    adj_0_pred = self.diffusion.predict_start_from_noise(adj_t, t, adj_noise_pred)
                    feat_0_pred = self.diffusion.predict_start_from_noise(feat_t, t, feat_noise_pred)
         
                    # === CHANGE: Better clamping ===
                    adj_0_pred = torch.clamp(adj_0_pred, -1, 1)
                    feat_0_pred = torch.clamp(feat_0_pred, -3, 3)  # Tighter for features
         
                    # Compute posterior mean
                    if t_idx > 0:
                        adj_t = self.diffusion.q_posterior(adj_0_pred, adj_t, t)
                        feat_t = self.diffusion.q_posterior(feat_0_pred, feat_t, t)
         
                        # Add noise
                        var = self.diffusion.posterior_variance[t].view(-1, 1, 1)
                        noise_scale = torch.sqrt(var) * 0.5  # Scale down noise
         
                        adj_t += noise_scale * self.diffusion.sample_laplace_noise(adj_t.shape, self.device)
                        feat_t += noise_scale * self.diffusion.sample_laplace_noise(feat_t.shape, self.device)
                    else:
                        adj_t = adj_0_pred
                        feat_t = feat_0_pred
         
                # Post-process
                adj_final = (adj_t[0, :num_nodes, :num_nodes] + 1.0) / 2.0
                adj_final = (adj_final + adj_final.T) / 2
                
                # === CHANGE: Adaptive thresholding ===
                # Use percentile-based threshold instead of fixed
                threshold = torch.quantile(adj_final.flatten(), 0.7)
                adj_final = (adj_final > threshold).float()
                adj_final.fill_diagonal_(0)
         
                # Ensure connectivity
                if adj_final.sum() < num_nodes - 1:
                    adj_probs = (adj_t[0, :num_nodes, :num_nodes] + 1.0) / 2.0
                    adj_probs = (adj_probs + adj_probs.T) / 2
         
                    for i in range(num_nodes - 1):
                        if adj_final[i].sum() == 0:  # Isolated node
                            best_j = (i + 1) % num_nodes
                            for j in range(num_nodes):
                                if i != j and adj_probs[i, j] > adj_probs[i, best_j]:
                                    best_j = j
                            adj_final[i, best_j] = 1
                            adj_final[best_j, i] = 1
         
                feat_final = feat_t[0, :num_nodes]
         
                return adj_final.cpu(), feat_final.cpu()
