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

def train_model(model, dataloader, num_epochs=150, lr=1e-3, beta_schedule='constant'):
            """Training loop with beta annealing for KL term"""
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-5
            )

            print(f"\nTraining for {num_epochs} epochs...")
            print(f"Beta schedule: {beta_schedule}")

            for epoch in range(num_epochs):
                # Beta annealing (cyclical or linear)
                if beta_schedule == 'linear':
                    beta = min(1.0, epoch / (num_epochs * 0.5))
                elif beta_schedule == 'cyclical':
                    cycle = epoch % (num_epochs // 4)
                    beta = min(1.0, cycle / (num_epochs // 8))
                else:  # constant
                    beta = 1.0

                total_loss = total_adj = total_feat = total_kl = total_sparse = 0.0
                count = 0

                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
                for adj, feat, labels, mask in pbar:
                    adj = adj.to(device)
                    feat = feat.to(device)
                    labels = labels.to(device)
                    mask = mask.to(device)

                    loss, adj_l, feat_l, kl_l, sparse_l = model.train_step(
                        adj, feat, labels, mask, optimizer, beta
                    )

                    if loss > 0:
                        total_loss += loss
                        total_adj += adj_l
                        total_feat += feat_l
                        total_kl += kl_l
                        total_sparse += sparse_l
                        count += 1

                        pbar.set_postfix({'loss': f'{loss:.4f}', 'beta': f'{beta:.3f}'})

                scheduler.step()

                if count > 0:
                    avg_loss = total_loss / count
                    avg_adj = total_adj / count
                    avg_feat = total_feat / count
                    avg_kl = total_kl / count
                    avg_sparse = total_sparse / count

                    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} "
                          f"(Adj: {avg_adj:.4f}, Feat: {avg_feat:.4f}, KL: {avg_kl:.4f}, "
                          f"Sparse: {avg_sparse:.4f}) | Beta: {beta:.3f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")

            print("\n✓ Training completed!")