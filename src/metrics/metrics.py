
import os
import random
import torch
import argparse
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

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde   

import config

#from src.models.gnn import GraphEncoder
class GCNEncoder(nn.Module):
    def __init__(self, node_features, hidden_dim=256, embedding_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
 
    def forward(self, x, edge_index, batch):
        h = F.relu(self.norm1(self.conv1(x, edge_index)))
        h = F.relu(self.norm2(self.conv2(h, edge_index)))
        return global_mean_pool(self.conv3(h, edge_index), batch)
 
 
def compute_fgd(orig_list, gen_list, encoder, device="cpu"):
    encoder = encoder.to(device).eval()
    scores  = []
    for og, gg in zip(orig_list, gen_list):
        og = og.to(device); gg = gg.to(device)
        ob = torch.zeros(og.x.size(0), dtype=torch.long, device=device)
        gb = torch.zeros(gg.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            oe = encoder(og.x, og.edge_index, ob)
            ge = encoder(gg.x, gg.edge_index, gb)
        def _stats(e):
            if e.dim() == 1: e = e.unsqueeze(0)
            mu = e.mean(0); d = e - mu
            cov = (torch.eye(e.size(1), device=e.device) * 1e-3
                   if e.size(0) == 1
                   else d.t().mm(d) / (e.size(0) - 1 + 1e-8))
            return mu, cov
        mu1, c1 = _stats(oe); mu2, c2 = _stats(ge)
        eps = 1e-6
        c1  += torch.eye(c1.size(0), device=c1.device) * eps
        c2  += torch.eye(c2.size(0), device=c2.device) * eps
        try:
            sq = torch.linalg.cholesky((c1 @ c2).float())
        except RuntimeError:
            sq = torch.eye(c1.size(0), device=c1.device) * eps
        fd = (mu1 - mu2).dot(mu1 - mu2) + torch.trace(c1 + c2 - 2 * sq)
        scores.append(fd.item())
    return scores
 
 
def pyg_to_grakel(data):
    if data.edge_index.size(1) == 0:
        return None
    G  = to_networkx(data, to_undirected=True)
    nl = {n: (int(data.x[n].argmax()) if data.x.size(1) > 1 else 0) for n in G.nodes()}
    return GrakelGraph(list(G.edges()), node_labels=nl)
 
 
def compute_metrics(orig_list, gen_list, node_features, device="cpu"):
    encoder = GCNEncoder(node_features).to(device).eval()
    euclid_l, mmd_l, kernel_l, spectral_l = [], [], [], []
    for og, gg in zip(orig_list, gen_list):
        og = og.to(device); gg = gg.to(device)
        ob = torch.zeros(og.x.size(0), dtype=torch.long, device=device)
        gb = torch.zeros(gg.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            eo = encoder(og.x, og.edge_index, ob).squeeze()
            eg = encoder(gg.x, gg.edge_index, gb).squeeze()
        # Euclidean
        euclid_l.append(float(torch.norm(eo - eg, p=2)))
        # MMD
        def gk(a, b, s=1.0):
            return torch.exp(-torch.norm(a - b, p=2).pow(2) / (2 * s ** 2))
        mmd_l.append(float(gk(eo, eo) + gk(eg, eg) - 2 * gk(eo, eg)))
        # WL kernel
        g1 = pyg_to_grakel(og); g2 = pyg_to_grakel(gg)
        if g1 and g2:
            try:
                K = WeisfeilerLehman(n_iter=2, base_graph_kernel=VertexHistogram).fit_transform([g1, g2])
                kernel_l.append(float(K[0, 1]))
            except Exception:
                kernel_l.append(0.0)
        else:
            kernel_l.append(0.0)
        # Spectral
        def spectrum(data, k=20):
            G = to_networkx(data, to_undirected=True)
            L = nx.normalized_laplacian_matrix(G).todense()
            ev = np.sort(np.real(np.linalg.eigvals(L)))
            return ev[:k] if len(ev) >= k else np.pad(ev, (0, k - len(ev)))
        spectral_l.append(float(np.linalg.norm(spectrum(og) - spectrum(gg))))
    return euclid_l, mmd_l, kernel_l, spectral_l
 
 
"""def compute_structural(orig_list, gen_list):
    def _ensure_nx(g):
        return to_networkx(g, to_undirected=True) if not isinstance(g, nx.Graph) else g
    def _ws(a, b):
        m = min(len(a), len(b))
        return (1 - wasserstein_distance(a[:m], b[:m])) * 100
    deg_s, lcc_s, bc_s, cc_s = [], [], [], []
    for og, gg in zip(orig_list, gen_list):
        go = _ensure_nx(og); gn = _ensure_nx(gg)
        deg_s.append(_ws([d for _, d in go.degree()], [d for _, d in gn.degree()]))
        lcc_s.append(_ws(list(nx.clustering(go).values()), list(nx.clustering(gn).values())))
        bc_s.append(_ws(list(nx.betweenness_centrality(go).values()),
                        list(nx.betweenness_centrality(gn).values())))
        cc_s.append(_ws(list(nx.closeness_centrality(go).values()),
                        list(nx.closeness_centrality(gn).values())))
    return deg_s, lcc_s, bc_s, cc_s"""

def compute_structural(orig_list, gen_list):

    def ensure_nx(G):
        return G if isinstance(G, nx.Graph) else to_networkx(G, to_undirected=True)

    def similarity_score(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if a.size == 0 or b.size == 0:
            return 0.0

        dist = wasserstein_distance(a, b)
        combined = np.concatenate([a, b])
        val_range = combined.max() - combined.min()

        if val_range == 0:
            return 100.0

        return max(0.0, (1 - dist / val_range) * 100)

    clust_s, bc_s, cc_s =  [], [], []

    for og, gg in zip(orig_list, gen_list):
        Go, Gg = ensure_nx(og), ensure_nx(gg)

        """deg_s.append(similarity_score(
            [d for _, d in Go.degree()],
            [d for _, d in Gg.degree()]
        ))"""

        clust_s.append(similarity_score(
            list(nx.clustering(Go).values()),
            list(nx.clustering(Gg).values())
        ))

        bc_s.append(similarity_score(
            list(nx.betweenness_centrality(Go).values()),
            list(nx.betweenness_centrality(Gg).values())
        ))

        cc_s.append(similarity_score(
            list(nx.closeness_centrality(Go).values()),
            list(nx.closeness_centrality(Gg).values())
        ))

    return clust_s, bc_s, cc_s


