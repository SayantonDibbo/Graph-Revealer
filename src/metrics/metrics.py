
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
class GraphEncoder(nn.Module):
    def __init__(self, node_features,
                 hidden_dim=config.ENCODER_HIDDEN_DIM,
                 embedding_dim=config.ENCODER_EMBED_DIM):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.norm1(self.conv1(x, edge_index)))
        h = F.relu(self.norm2(self.conv2(h, edge_index)))
        h = self.conv3(h, edge_index)
        return global_mean_pool(h, batch)

def compute_mean_and_cov(embeddings):
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    mean = embeddings.mean(dim=0)
    diff = embeddings - mean
    if embeddings.size(0) == 1:
        cov = torch.eye(embeddings.size(1), device=embeddings.device) * 1e-3
    else:
        cov = diff.t().mm(diff) / (embeddings.size(0) - 1 + 1e-8)
    return mean, cov


def frechet_distance(mu1, cov1, mu2, cov2, eps=config.ENCODER_EPS):
    diff = mu1 - mu2
    c1   = cov1 + torch.eye(cov1.size(0), device=cov1.device) * eps
    c2   = cov2 + torch.eye(cov2.size(0), device=cov2.device) * eps
    try:
        sqrt_cp = torch.linalg.cholesky((c1 @ c2).float())
    except RuntimeError:
        sqrt_cp = torch.eye(c1.size(0), device=c1.device) * eps
    return (diff.dot(diff) + torch.trace(c1 + c2 - 2 * sqrt_cp)).item()


def compute_fgd_scores(original_graphs, generated_graphs, encoder, device="cpu"):
    assert len(original_graphs) == len(generated_graphs)
    encoder = encoder.to(device); encoder.eval()
    scores  = []
    for og, gg in zip(original_graphs, generated_graphs):
        og = og.to(device); gg = gg.to(device)
        ob = torch.zeros(og.x.size(0), dtype=torch.long, device=device)
        gb = torch.zeros(gg.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            oe = encoder(og.x, og.edge_index, ob)
            ge = encoder(gg.x, gg.edge_index, gb)
        mu1, c1 = compute_mean_and_cov(oe)
        mu2, c2 = compute_mean_and_cov(ge)
        scores.append(frechet_distance(mu1, c1, mu2, c2))
    return scores, float(np.mean(scores)), float(np.std(scores))


def pyg_to_grakel(data):
    if data.edge_index.size(1) == 0:
        return None
    G_nx = to_networkx(data, to_undirected=True)
    node_labels = {
        n: int(data.x[n].argmax()) if data.x.size(1) > 1 else 0
        for n in G_nx.nodes()}
    return GrakelGraph(list(G_nx.edges()), node_labels=node_labels)


def euclidean_distance(e1, e2):
    return float(torch.norm(e1 - e2, p=2))


def mmd_distance(e1, e2, sigma=1.0):
    def k(x, y): return torch.exp(-torch.norm(x - y, p=2) ** 2 / (2 * sigma ** 2))
    return float(k(e1, e1) + k(e2, e2) - 2 * k(e1, e2))


def graph_kernel_similarity(og, gg, h=config.WL_ITERATIONS):
    g1, g2 = pyg_to_grakel(og), pyg_to_grakel(gg)
    if g1 is None or g2 is None:
        return 0.0
    K = WeisfeilerLehman(n_iter=h, base_graph_kernel=VertexHistogram).fit_transform([g1, g2])
    return float(K[0, 1])


def spectral_distance(og, gg, k=config.SPECTRAL_K):
    def spectrum(data):
        G = to_networkx(data, to_undirected=True)
        L = nx.normalized_laplacian_matrix(G).todense()
        ev = np.sort(np.real(np.linalg.eigvals(L)))
        return ev[:k] if len(ev) >= k else np.pad(ev, (0, k - len(ev)))
    return float(np.linalg.norm(spectrum(og) - spectrum(gg)))


def compute_graph_distances_list(original_graph, generated_graph, node_features, device="cpu"):
    assert len(original_graph) == len(generated_graph)
    encoder = GraphEncoder(node_features=node_features).to(device); encoder.eval()

    euclid_list = []; mmd_list = []; kernel_list = []; spectral_list = []
    for og, gg in zip(original_graph, generated_graph):
        og = og.to(device); gg = gg.to(device)
        ob = torch.zeros(og.x.size(0), dtype=torch.long, device=device)
        gb = torch.zeros(gg.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            eo = encoder(og.x, og.edge_index, ob).squeeze()
            eg = encoder(gg.x, gg.edge_index, gb).squeeze()
        euclid_list.append(euclidean_distance(eo, eg))
        mmd_list.append(mmd_distance(eo, eg))
        kernel_list.append(graph_kernel_similarity(og, gg))
        spectral_list.append(spectral_distance(og, gg))

    return euclid_list, mmd_list, kernel_list, spectral_list


def ensure_nx(G):
    return G if isinstance(G, nx.Graph) else to_networkx(G, to_undirected=True)


def similarity_score(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dist      = wasserstein_distance(a, b)
    combined  = np.concatenate([a, b])
    val_range = combined.max() - combined.min()
    return 100.0 if val_range == 0 else max(0.0, (1 - dist / val_range) * 100)


def compute_structural_scores(original_graph, generated_graph):
    deg_s = []; clust_s = []; bc_s = []; cc_s = []
    for og, gg in zip(original_graph, generated_graph):
        Go, Gg = ensure_nx(og), ensure_nx(gg)
        deg_s.append(similarity_score(
            [d for _, d in Go.degree()], [d for _, d in Gg.degree()]))
        clust_s.append(similarity_score(
            list(nx.clustering(Go).values()), list(nx.clustering(Gg).values())))
        bc_s.append(similarity_score(
            list(nx.betweenness_centrality(Go).values()),
            list(nx.betweenness_centrality(Gg).values())))
        cc_s.append(similarity_score(
            list(nx.closeness_centrality(Go).values()),
            list(nx.closeness_centrality(Gg).values())))
    return deg_s, clust_s, bc_s, cc_s




   


