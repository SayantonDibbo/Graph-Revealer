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


def train_gan(gan, dataloader, num_epochs, lr_g, lr_d):
    opt_G = torch.optim.Adam(gan.generator.parameters(),    lr=lr_g, betas=(0.5, 0.999), weight_decay=1e-5)
    opt_D = torch.optim.Adam(gan.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999), weight_decay=1e-5)
    sch_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=num_epochs, eta_min=1e-6)
    sch_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=num_epochs, eta_min=1e-6)
    for epoch in range(num_epochs):
        td, tg, cnt = 0.0, 0.0, 0
        for adj, feat, labels, mask in tqdm(dataloader,
                                            desc=f"GAN epoch {epoch+1}/{num_epochs}",
                                            leave=False):
            adj    = adj.to(gan.device);    feat   = feat.to(gan.device)
            labels = labels.to(gan.device); mask   = mask.to(gan.device)
            dl, gl = gan.train_step(adj, feat, labels, mask, opt_G, opt_D)
            td += dl; tg += gl; cnt += 1
        sch_G.step(); sch_D.step()
        if cnt:
            print(f"  GAN [{epoch+1}/{num_epochs}] D={td/cnt:.4f}  G={tg/cnt:.4f}")
    print("✓ GAN training complete")