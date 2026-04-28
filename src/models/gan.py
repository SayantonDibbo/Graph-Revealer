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
                def __init__(self, data, max_nodes=126):
                    print("Loading NCI1 dataset...")
                    #self.dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
                    self.datasets = data
                    self.max_nodes = max_nodes
                    self.num_classes = classes
        
                    # Filter and validate graphs
                    self.valid_data = []
                    for data in self.datasets:
                        if (data.x is not None and data.x.shape[1] > 0 and 
                            8 <= data.num_nodes <= max_nodes and
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
        
        
        class Generator(nn.Module):
            """Generator network for graph generation"""
            def __init__(self, latent_dim, max_nodes, feature_dim, num_classes, hidden_dim=256):
                super().__init__()
                self.max_nodes = max_nodes
                self.feature_dim = feature_dim
                self.latent_dim = latent_dim
        
                # Label embedding
                self.label_embed = nn.Embedding(num_classes, hidden_dim)
        
                # Input projection
                self.input_proj = nn.Sequential(
                    nn.Linear(latent_dim + hidden_dim, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim * 4),
                    nn.BatchNorm1d(hidden_dim * 4),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                )
        
                # Adjacency generator
                self.adj_generator = nn.Sequential(
                    nn.Linear(hidden_dim * 4, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim * 2, max_nodes * max_nodes),
                    nn.Sigmoid()  # Output in [0, 1]
                )
        
                # Feature generator
                self.feat_generator = nn.Sequential(
                    nn.Linear(hidden_dim * 4, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim * 2, max_nodes * feature_dim),
                    nn.Tanh()  # Output in [-1, 1]
                )
        
            def forward(self, z, labels):
                """
                Args:
                    z: [B, latent_dim] noise vector
                    labels: [B] class labels
                Returns:
                    adj: [B, N, N] adjacency matrix
                    features: [B, N, F] node features
                """
                B = z.shape[0]
        
                # Embed labels
                label_emb = self.label_embed(labels)  # [B, hidden_dim]
        
                # Concatenate noise and label
                x = torch.cat([z, label_emb], dim=1)  # [B, latent_dim + hidden_dim]
        
                # Project
                h = self.input_proj(x)  # [B, hidden_dim * 4]
        
                # Generate adjacency
                adj = self.adj_generator(h)  # [B, N*N]
                adj = adj.view(B, self.max_nodes, self.max_nodes)
        
                # Make symmetric
                adj = (adj + adj.transpose(1, 2)) / 2
                adj = adj * (1 - torch.eye(self.max_nodes, device=adj.device).unsqueeze(0))
        
                # Generate features
                feat = self.feat_generator(h)  # [B, N*F]
                feat = feat.view(B, self.max_nodes, self.feature_dim)
        
                return adj, feat
        
        
        class Discriminator(nn.Module):
            """Discriminator network for graph classification"""
            def __init__(self, max_nodes, feature_dim, num_classes, hidden_dim=256):
                super().__init__()
                self.max_nodes = max_nodes
                self.feature_dim = feature_dim
        
                # Label embedding
                self.label_embed = nn.Embedding(num_classes, hidden_dim)
        
                # Adjacency encoder
                self.adj_encoder = nn.Sequential(
                    nn.Linear(max_nodes, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2)
                )
        
                # Feature encoder
                self.feat_encoder = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2)
                )
        
                # Graph-level processing
                self.graph_processor = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                )
        
                # Output layer
                self.output = nn.Sequential(
                    nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 1)
                )
        
            def forward(self, adj, features, labels, mask):
                """
                Args:
                    adj: [B, N, N] adjacency matrix
                    features: [B, N, F] node features
                    labels: [B] class labels
                    mask: [B, N] node mask
                Returns:
                    validity: [B, 1] real/fake score
                """
                B, N = adj.shape[:2]
        
                # Encode
                h_adj = self.adj_encoder(adj)  # [B, N, hidden_dim]
                h_feat = self.feat_encoder(features)  # [B, N, hidden_dim]
        
                # Concatenate
                h = torch.cat([h_adj, h_feat], dim=-1)  # [B, N, hidden_dim*2]
        
                # Apply mask and pool
                h = h * mask.unsqueeze(-1)
                h = self.graph_processor(h)  # [B, N, hidden_dim]
        
                # Global pooling (mean)
                graph_emb = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        
                # Condition on label
                label_emb = self.label_embed(labels)  # [B, hidden_dim]
        
                # Combine
                combined = torch.cat([graph_emb, label_emb], dim=1)  # [B, hidden_dim*2]
        
                # Output
                validity = self.output(combined)  # [B, 1]
        
                return validity
        
        
        class GraphGAN:
            """GAN model for graph generation"""
            def __init__(self, latent_dim, max_nodes, feature_dim, num_classes, device='cpu'):
                self.device = device
                self.latent_dim = latent_dim
                self.max_nodes = max_nodes
                self.feature_dim = feature_dim
                self.num_classes = num_classes
        
                # Initialize networks
                self.generator = Generator(latent_dim, max_nodes, feature_dim, num_classes).to(device)
                self.discriminator = Discriminator(max_nodes, feature_dim, num_classes).to(device)
        
                # Loss function
                self.adversarial_loss = nn.BCEWithLogitsLoss()
        
            def train_step(self, real_adj, real_feat, labels, mask, 
                           optimizer_G, optimizer_D):
                """Training step for GAN"""
                B = real_adj.shape[0]
        
                # Labels for real and fake
                real_labels = torch.ones(B, 1, device=self.device)
                fake_labels = torch.zeros(B, 1, device=self.device)
        
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
        
                # Real graphs
                real_validity = self.discriminator(real_adj, real_feat, labels, mask)
                d_real_loss = self.adversarial_loss(real_validity, real_labels)
        
                # Fake graphs
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_adj, fake_feat = self.generator(z, labels)
        
                # Apply mask to fake graphs
                fake_adj = fake_adj * mask.unsqueeze(-1) * mask.unsqueeze(1)
                fake_feat = fake_feat * mask.unsqueeze(-1)
        
                fake_validity = self.discriminator(fake_adj.detach(), fake_feat.detach(), 
                                                   labels, mask)
                d_fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
        
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                optimizer_D.step()
        
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
        
                # Generate new batch
                z = torch.randn(B, self.latent_dim, device=self.device)
                gen_adj, gen_feat = self.generator(z, labels)
        
                # Apply mask
                gen_adj = gen_adj * mask.unsqueeze(-1) * mask.unsqueeze(1)
                gen_feat = gen_feat * mask.unsqueeze(-1)
        
                # Validity check
                validity = self.discriminator(gen_adj, gen_feat, labels, mask)
                g_loss = self.adversarial_loss(validity, real_labels)
        
                # Add sparsity regularization
                mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(1)
                sparsity_loss = (gen_adj * mask_2d).mean()
        
                # Add feature regularization (keep features reasonable)
                feat_reg = ((gen_feat * mask.unsqueeze(-1)) ** 2).mean()
        
                # Total generator loss
                total_g_loss = g_loss + 0.1 * sparsity_loss + 0.05 * feat_reg
        
                total_g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                optimizer_G.step()
        
                return d_loss.item(), g_loss.item(), d_real_loss.item(), d_fake_loss.item()
        
            @torch.no_grad()
            def sample(self, label, num_nodes=None):
                """Generate graph from noise"""
                self.generator.eval()
        
                if num_nodes is None:
                    num_nodes = np.random.randint(15, min(25, self.max_nodes))
                num_nodes = min(num_nodes, self.max_nodes)
        
                # Sample noise
                z = torch.randn(1, self.latent_dim, device=self.device)
                label_tensor = torch.tensor([label], dtype=torch.long, device=self.device)
        
                # Generate
                adj, feat = self.generator(z, label_tensor)
        
                # Extract relevant nodes
                adj_final = adj[0, :num_nodes, :num_nodes]
                feat_final = feat[0, :num_nodes]
        
                # Post-process adjacency for sparsity
                # Threshold to create sparse molecular-like graphs
                threshold = 0.6  # Adjust for desired sparsity
                adj_final = (adj_final > threshold).float()
        
                # Ensure symmetry and no self-loops
                adj_final = (adj_final + adj_final.T) / 2
                adj_final = (adj_final > 0.5).float()
                adj_final.fill_diagonal_(0)
        
                # Ensure minimum connectivity
                if adj_final.sum() < num_nodes - 1:
                    # Create minimum spanning tree
                    for i in range(num_nodes - 1):
                        if adj_final[i:, :].sum() == 0:  # Isolated node
                            # Connect to closest node
                            j = (i + 1) % num_nodes
                            adj_final[i, j] = 1
                            adj_final[j, i] = 1
        
                self.generator.train()
                return adj_final.cpu(), feat_final.cpu()