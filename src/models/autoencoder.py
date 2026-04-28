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

class NCI1GraphDataset(Dataset):
            """NCI1 dataset for Graph AE"""
            def __init__(self, data, max_nodes=126):
                print("Loading NCI1 dataset...")
                self.dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
                self.datasets = data
                self.max_nodes = max_nodes
                self.num_classes = self.dataset.num_classes

                # Filter and validate graphs
                self.valid_data = []
                for data in self.datasets:
                    if (data.x is not None and data.x.shape[1] > 0 and
                        8 <= data.num_nodes <= max_nodes and
                        data.edge_index.shape[1] > 0):
                        avg_degree = data.edge_index.shape[1] / data.num_nodes
                        if avg_degree <= 5.0:
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
                    avg_edges = np.mean([d.edge_index.shape[1] / 2 for d in class_graphs])
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

                # Dense adjacency
                adj = to_dense_adj(data.edge_index, max_num_nodes=self.max_nodes)[0]

                # Pad features
                features = torch.zeros(self.max_nodes, self.feature_dim)
                features[:data.num_nodes] = features_norm

                # Create mask
                mask = torch.zeros(self.max_nodes)
                mask[:data.num_nodes] = 1

                return adj, features, data.y.item(), mask


        class GraphEncoder(nn.Module):
            """Encoder: Graph -> Latent distribution parameters (mu, logvar)"""
            def __init__(self, max_nodes, feature_dim, num_classes, hidden_dim=256, latent_dim=128):
                super().__init__()
                self.max_nodes = max_nodes
                self.feature_dim = feature_dim
                self.latent_dim = latent_dim

                # Label embedding
                self.label_embed = nn.Embedding(num_classes, hidden_dim)

                # Adjacency encoder
                self.adj_encoder = nn.Sequential(
                    nn.Linear(max_nodes, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                )

                # Feature encoder
                self.feat_encoder = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                )

                # Joint processing with label conditioning
                self.joint_processor = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                )

                # Global pooling
                self.pool = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )

                # Latent parameters
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            def forward(self, adj, features, label, mask):
                """
                Args:
                    adj: [B, N, N]
                    features: [B, N, F]
                    label: [B]
                    mask: [B, N]
                Returns:
                    mu: [B, latent_dim]
                    logvar: [B, latent_dim]
                """
                B, N = adj.shape[:2]

                # Label conditioning
                label_emb = self.label_embed(label)  # [B, hidden]

                # Encode adjacency and features
                h_adj = self.adj_encoder(adj)  # [B, N, hidden]
                h_feat = self.feat_encoder(features)  # [B, N, hidden]

                # Concatenate
                h = torch.cat([h_adj, h_feat], dim=-1)  # [B, N, 2*hidden]

                # Process
                h = self.joint_processor(h)  # [B, N, hidden]

                # Apply mask
                h = h * mask.unsqueeze(-1)

                # Global pooling (mean over nodes)
                h_global = (h.sum(dim=1) / mask.sum(dim=1, keepdim=True))  # [B, hidden]

                # Add label conditioning
                h_global = self.pool(h_global + label_emb)  # [B, hidden]

                # Latent parameters
                mu = self.fc_mu(h_global)  # [B, latent_dim]
                logvar = self.fc_logvar(h_global)  # [B, latent_dim]

                return mu, logvar


        class GraphDecoder(nn.Module):
            """Decoder: Latent + Label -> Graph (adjacency + features)"""
            def __init__(self, max_nodes, feature_dim, num_classes, hidden_dim=256, latent_dim=128):
                super().__init__()
                self.max_nodes = max_nodes
                self.feature_dim = feature_dim

                # Label embedding
                self.label_embed = nn.Embedding(num_classes, hidden_dim)

                # Latent to hidden
                self.latent_proj = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                )

                # Expand to node-level
                self.node_expander = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * max_nodes),
                    nn.LayerNorm(hidden_dim * max_nodes),
                    nn.ReLU()
                )

                # Separate decoders for adjacency and features
                self.adj_decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, max_nodes)
                )

                self.feat_decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, feature_dim)
                )

            def forward(self, z, label, mask):
                """
                Args:
                    z: [B, latent_dim]
                    label: [B]
                    mask: [B, N]
                Returns:
                    adj_logits: [B, N, N]
                    feat_recon: [B, N, F]
                """
                B = z.shape[0]
                N = self.max_nodes

                # Label conditioning
                label_emb = self.label_embed(label)  # [B, hidden]

                # Project latent
                h = self.latent_proj(z) + label_emb  # [B, hidden]

                # Expand to nodes
                h = self.node_expander(h)  # [B, hidden * N]
                h = h.view(B, N, -1)  # [B, N, hidden]

                # Apply mask
                h = h * mask.unsqueeze(-1)

                # Decode adjacency (using outer product idea)
                adj_logits = self.adj_decoder(h)  # [B, N, N]

                # Make symmetric
                adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2

                # Decode features
                feat_recon = self.feat_decoder(h)  # [B, N, F]

                return adj_logits, feat_recon


        class GraphCVAE(nn.Module):
            """Conditional Variational Autoencoder for Graphs"""
            def __init__(self, max_nodes, feature_dim, num_classes, hidden_dim=256,
                         latent_dim=128, device='cpu'):
                super().__init__()
                self.device = device
                self.max_nodes = max_nodes
                self.feature_dim = feature_dim
                self.latent_dim = latent_dim

                self.encoder = GraphEncoder(max_nodes, feature_dim, num_classes,
                                            hidden_dim, latent_dim).to(device)
                self.decoder = GraphDecoder(max_nodes, feature_dim, num_classes,
                                            hidden_dim, latent_dim).to(device)

            def sample_laplace_noise(self, shape, device, epsilon=0.5):
                """Sample Laplace noise with epsilon parameter"""
                u = torch.rand(shape, device=device) - 0.5
                b = epsilon / np.sqrt(2)
                noise = -b * torch.sign(u) * torch.log(torch.clamp(1 - 2 * torch.abs(u), min=1e-10))
                return torch.clamp(noise, -10, 10)

            def reparameterize(self, mu, logvar):
                """Reparameterization trick with Laplace noise: z = mu + sigma * epsilon"""
                std = torch.exp(0.5 * logvar)
                eps = self.sample_laplace_noise(std.shape, std.device, epsilon=0.5)
                return mu + eps * std

            def forward(self, adj, features, label, mask):
                """
                Args:
                    adj: [B, N, N]
                    features: [B, N, F]
                    label: [B]
                    mask: [B, N]
                Returns:
                    adj_logits, feat_recon, mu, logvar
                """
                # Encode
                mu, logvar = self.encoder(adj, features, label, mask)

                # Reparameterize
                z = self.reparameterize(mu, logvar)

                # Decode
                adj_logits, feat_recon = self.decoder(z, label, mask)

                return adj_logits, feat_recon, mu, logvar

            def loss_function(self, adj, features, adj_logits, feat_recon, mu, logvar, mask, beta=1.0):
                """
                Compute VAE loss = Reconstruction Loss + KL Divergence

                Args:
                    beta: weight for KL divergence (beta-VAE)
                """
                B = adj.shape[0]

                # Masks
                mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(1)  # [B, N, N]
                mask_1d = mask.unsqueeze(-1)  # [B, N, 1]

                # Adjacency reconstruction loss (Binary Cross Entropy)
                adj_recon_loss = F.binary_cross_entropy_with_logits(
                    adj_logits * mask_2d,
                    adj * mask_2d,
                    reduction='sum'
                ) / mask_2d.sum()

                # Feature reconstruction loss (MSE)
                feat_recon_loss = F.mse_loss(
                    feat_recon * mask_1d,
                    features * mask_1d,
                    reduction='sum'
                ) / mask_1d.sum()

                # KL divergence: KL(q(z|x,y) || p(z))
                # Note: Using Gaussian approximation for KL with Laplace prior
                # For true Laplace KL, we'd need Monte Carlo estimation
                # This is an approximation that works reasonably well in practice
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

                # Sparsity regularization (L1 on edge probabilities)
                adj_probs = torch.sigmoid(adj_logits)
                sparsity_loss = (adj_probs * mask_2d).sum() / mask_2d.sum()

                # Total loss
                total_loss = adj_recon_loss + feat_recon_loss + beta * kl_loss + 0.05 * sparsity_loss

                return total_loss, adj_recon_loss, feat_recon_loss, kl_loss, sparsity_loss

            def train_step(self, adj, features, labels, mask, optimizer, beta=1.0):
                """Training step"""
                self.train()

                # Forward pass
                adj_logits, feat_recon, mu, logvar = self(adj, features, labels, mask)

                # Compute loss
                total_loss, adj_loss, feat_loss, kl_loss, sparse_loss = self.loss_function(
                    adj, features, adj_logits, feat_recon, mu, logvar, mask, beta
                )

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("⚠ NaN detected, skipping batch")
                    return 0.0, 0.0, 0.0, 0.0, 0.0

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                return (total_loss.item(), adj_loss.item(), feat_loss.item(),
                        kl_loss.item(), sparse_loss.item())

            @torch.no_grad()
            def sample(self, label, num_nodes=None):
                """Generate graph from random latent code using Laplace prior"""
                self.eval()

                if num_nodes is None:
                    num_nodes = np.random.randint(15, min(25, self.max_nodes))
                num_nodes = min(num_nodes, self.max_nodes)

                # Sample from Laplace prior instead of Gaussian
                z = self.sample_laplace_noise((1, self.latent_dim), self.device, epsilon=0.5)

                # Mask
                mask = torch.zeros(1, self.max_nodes, device=self.device)
                mask[0, :num_nodes] = 1

                # Label
                label_tensor = torch.tensor([label], dtype=torch.long, device=self.device)

                # Decode
                adj_logits, feat_recon = self.decoder(z, label_tensor, mask)

                # Post-process adjacency
                adj_probs = torch.sigmoid(adj_logits[0, :num_nodes, :num_nodes])
                adj_probs = (adj_probs + adj_probs.T) / 2  # Symmetrize

                # Threshold for sparse graphs
                threshold = 0.5
                adj_final = (adj_probs > threshold).float()
                adj_final.fill_diagonal_(0)

                # Ensure connectivity (minimum spanning tree)
                if adj_final.sum() < num_nodes - 1:
                    for i in range(num_nodes - 1):
                        best_j = i + 1
                        best_prob = adj_probs[i, best_j]
                        for j in range(i + 1, num_nodes):
                            if adj_probs[i, j] > best_prob and adj_final[i, j] == 0:
                                best_j = j
                                best_prob = adj_probs[i, j]

                        if adj_final[i, best_j] == 0:
                            adj_final[i, best_j] = 1
                            adj_final[best_j, i] = 1

                feat_final = feat_recon[0, :num_nodes]

                return adj_final.cpu(), feat_final.cpu()