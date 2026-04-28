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

## need model here and dataloader
def train_model(model, dataloader, num_epochs=150, lr=1e-3):
            """
            CHANGES:
            1. Better optimizer settings
            2. Warmup learning rate
            """
            optimizer = torch.optim.AdamW(
                model.model.parameters(), 
                lr=lr, 
                weight_decay=1e-4,
                betas=(0.9, 0.999)  # Standard settings
            )
         
            # === CHANGE: Added warmup ===
            warmup_epochs = 10
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    # Cosine decay after warmup
                    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
         
            print(f"\nTraining for {num_epochs} epochs...")
         
            for epoch in range(num_epochs):
                total_loss = total_adj = total_feat = 0.0
                count = 0
         
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
                for adj, feat, labels, mask in pbar:
                    adj = adj.to(model.device)
                    feat = feat.to(model.device)
                    labels = labels.to(model.device)
                    mask = mask.to(model.device)
         
                    loss, adj_l, feat_l = model.train_step(adj, feat, labels, mask, optimizer)
         
                    if loss > 0:
                        total_loss += loss
                        total_adj += adj_l
                        total_feat += feat_l
                        count += 1
         
                        pbar.set_postfix({'loss': f'{loss:.4f}'})
         
                scheduler.step()
         
                if count > 0:
                    avg_loss = total_loss / count
                    avg_adj = total_adj / count
                    avg_feat = total_feat / count
                    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} "
                          f"(Adj: {avg_adj:.6f}, Feat: {avg_feat:.6f}) | "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")
         
            print("\n✓ Training completed!")