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
import config
import argparse
## need model here and dataloader
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