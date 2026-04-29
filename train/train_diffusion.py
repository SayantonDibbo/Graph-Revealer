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
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 10
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        p = (ep - warmup) / max(num_epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for epoch in range(num_epochs):
        total_loss = 0.0; count = 0
        for adj, feat, labels, mask in tqdm(dataloader,
                                            desc=f"Diffusion epoch {epoch+1}/{num_epochs}",
                                            leave=False):
            adj    = adj.to(model.device);   feat   = feat.to(model.device)
            labels = labels.to(model.device); mask   = mask.to(model.device)
            l, _, _ = model.train_step(adj, feat, labels, mask, optimizer)
            if l > 0:
                total_loss += l; count += 1
        scheduler.step()
        if count:
            print(f"  Diffusion [{epoch+1}/{num_epochs}] loss={total_loss/count:.5f}")
    print("✓ Diffusion training complete")