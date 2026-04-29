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

def train_cvae(model, dataloader, num_epochs, lr, beta_schedule="linear"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    for epoch in range(num_epochs):
        if beta_schedule == "linear":
            beta = min(1.0, epoch / max(num_epochs * 0.5, 1))
        elif beta_schedule == "cyclical":
            beta = min(1.0, (epoch % max(num_epochs // 4, 1)) / max(num_epochs // 8, 1))
        else:
            beta = 1.0
        tl, cnt = 0.0, 0
        for adj, feat, labels, mask in tqdm(dataloader,
                                            desc=f"CVAE epoch {epoch+1}/{num_epochs}",
                                            leave=False):
            adj = adj.to(model.device); feat = feat.to(model.device)
            labels = labels.to(model.device); mask = mask.to(model.device)
            l, *_ = model.train_step(adj, feat, labels, mask, optimizer, beta)
            if l > 0:
                tl += l; cnt += 1
        scheduler.step()
        if cnt:
            print(f"  CVAE [{epoch+1}/{num_epochs}] loss={tl/cnt:.5f}  beta={beta:.3f}")
    print("✓ CVAE training complete")