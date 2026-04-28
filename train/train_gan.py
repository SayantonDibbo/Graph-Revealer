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

def train_gan(gan, dataloader, num_epochs=150, lr_g=2e-4, lr_d=2e-4):
            """Training loop for GAN"""
            optimizer_G = torch.optim.Adam(gan.generator.parameters(), lr=lr_g, 
                                           betas=(0.5, 0.999), weight_decay=1e-5)
            optimizer_D = torch.optim.Adam(gan.discriminator.parameters(), lr=lr_d, 
                                           betas=(0.5, 0.999), weight_decay=1e-5)
        
            # Learning rate schedulers
            scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_G, T_max=num_epochs, eta_min=1e-6
            )
            scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_D, T_max=num_epochs, eta_min=1e-6
            )
        
            print(f"\nTraining GAN for {num_epochs} epochs...")
        
            for epoch in range(num_epochs):
                total_d_loss = 0.0
                total_g_loss = 0.0
                total_d_real = 0.0
                total_d_fake = 0.0
                count = 0
        
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
                for adj, feat, labels, mask in pbar:
                    adj = adj.to(device)
                    feat = feat.to(device)
                    labels = labels.to(device)
                    mask = mask.to(device)
        
                    d_loss, g_loss, d_real, d_fake = gan.train_step(
                        adj, feat, labels, mask, optimizer_G, optimizer_D
                    )
        
                    total_d_loss += d_loss
                    total_g_loss += g_loss
                    total_d_real += d_real
                    total_d_fake += d_fake
                    count += 1
        
                    pbar.set_postfix({
                        'D_loss': f'{d_loss:.4f}',
                        'G_loss': f'{g_loss:.4f}'
                    })
        
                scheduler_G.step()
                scheduler_D.step()
        
                if count > 0:
                    avg_d = total_d_loss / count
                    avg_g = total_g_loss / count
                    avg_d_real = total_d_real / count
                    avg_d_fake = total_d_fake / count
        
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                          f"D_loss: {avg_d:.6f} (Real: {avg_d_real:.6f}, Fake: {avg_d_fake:.6f}) | "
                          f"G_loss: {avg_g:.6f} | "
                          f"LR_G: {scheduler_G.get_last_lr()[0]:.6f}, "
                          f"LR_D: {scheduler_D.get_last_lr()[0]:.6f}")
        
            print("\n✓ Training completed!")