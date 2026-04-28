
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

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde    

#from src.models.gnn import GraphEncoder

class GraphEncoder(nn.Module):
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
        h = self.conv3(h, edge_index)
        graph_embedding = global_mean_pool(h, batch)
        return graph_embedding

# take in original_graph and generated_graph 
encoder = GraphEncoder(node_features=node_features)
     
def compute_mean_and_cov(embeddings):
    """
    embeddings: Tensor [1, D] or [N, D]
    """
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)

    mean = embeddings.mean(dim=0)
    diff = embeddings - mean

    if embeddings.size(0) == 1:
        cov = torch.eye(embeddings.size(1), device=embeddings.device) * 1e-3
    else:
        cov = diff.t().mm(diff) / (embeddings.size(0) - 1 + 1e-8)

    return mean, cov


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    diff = mu1 - mu2

    cov1 = cov1 + torch.eye(cov1.size(0), device=cov1.device) * eps
    cov2 = cov2 + torch.eye(cov2.size(0), device=cov2.device) * eps

    try:
        cov_prod = (cov1 @ cov2).float()
        sqrt_cov_prod = torch.linalg.cholesky(cov_prod)
    except RuntimeError:
        sqrt_cov_prod = torch.eye(cov1.size(0), device=cov1.device) * eps

    fd = diff.dot(diff) + torch.trace(cov1 + cov2 - 2 * sqrt_cov_prod)
    return fd.item()

# Frechet Graph Distance (FGD)
def compute_fgd_scores(original_graphs, generated_graphs, encoder, device="cpu"):
    """
    original_graphs: list of PyG Data objects
    generated_graphs: list of PyG Data objects
    encoder: GraphEncoder
    """

    assert len(original_graphs) == len(generated_graphs), \
        "Number of original and generated graphs must match."

    encoder = encoder.to(device)
    encoder.eval()

    fgd_scores = []

    for o_graph, g_graph in zip(original_graphs, generated_graphs):

        # Move graphs to device
        o_graph = o_graph.to(device)
        g_graph = g_graph.to(device)

        # Build batch vectors manually (each graph = batch=0)
        o_batch = torch.zeros(o_graph.x.size(0), dtype=torch.long, device=device)
        g_batch = torch.zeros(g_graph.x.size(0), dtype=torch.long, device=device)

        # Encode embeddings
        with torch.no_grad():
            o_emb = encoder(o_graph.x, o_graph.edge_index, o_batch)
            g_emb = encoder(g_graph.x, g_graph.edge_index, g_batch)

        # Compute stats
        mu1, cov1 = compute_mean_and_cov(o_emb)
        mu2, cov2 = compute_mean_and_cov(g_emb)

        # FGD per graph-pair
        fgd = frechet_distance(mu1, cov1, mu2, cov2)
        fgd_scores.append(fgd)

    # Stats
    fgd_mean = float(np.mean(fgd_scores))
    fgd_std = float(np.std(fgd_scores))

    return fgd_scores, fgd_mean, fgd_std


# Convert PyG Data → GraKeL Graph
def pyg_to_grakel_graph(data):
    if data.edge_index.size(1) == 0:
        return None
    G_nx = to_networkx(data, to_undirected=True)

    # Simple node labeling: argmax of feature vector
    node_labels = {
        n: int(data.x[n].argmax()) if data.x.size(1) > 1 else 0
        for n in G_nx.nodes()
    }
    edges = list(G_nx.edges())

    return GrakelGraph(edges, node_labels=node_labels)



# Euclidean Distance (embeddings)

emb1 = original_graph
emb2 = generated_graph
def euclidean_distance(emb1, emb2):
    return float(torch.norm(emb1 - emb2, p=2))



# Maximum Mean Discrepancy (MMD)

def mmd_distance(emb1, emb2, sigma=1.0):
    def gaussian_kernel(x, y, sig):
        return torch.exp(-torch.norm(x - y, p=2)**2 / (2 * sig**2))

    return float(
        gaussian_kernel(emb1, emb1, sigma)
        + gaussian_kernel(emb2, emb2, sigma)
        - 2 * gaussian_kernel(emb1, emb2, sigma)
    )

# Graph Kernel Similarity (WL)
def graph_kernel_similarity(original_graph, generated_graph, h=2):
    g1 = pyg_to_grakel_graph(original_graph)
    g2 = pyg_to_grakel_graph(generated_graph)

    wl_kernel = WeisfeilerLehman(n_iter=h, base_graph_kernel=VertexHistogram)
    K = wl_kernel.fit_transform([g1, g2])

    return float(K[0, 1])

# Spectral Distance (Laplacian)
def spectral_distance(original_graph, generated_graph, k=20):
    def laplacian_spectrum(data, k):
        G = to_networkx(data, to_undirected=True)
        L = nx.normalized_laplacian_matrix(G).todense()
        eigvals = np.sort(np.real(np.linalg.eigvals(L)))
        if len(eigvals) >= k:
            return eigvals[:k]
        return np.pad(eigvals, (0, k - len(eigvals)))

    s1 = laplacian_spectrum(original_graph, k)
    s2 = laplacian_spectrum(generated_graph, k)
    return float(np.linalg.norm(s1 - s2))

  
    
def ensure_nx(G):
    if not isinstance(G, nx.Graph):
        G = to_networkx(G, to_undirected=True)
    return G


def degree_distribution(G):
    G = ensure_nx(G)
    return np.array([d for _, d in G.degree()])


def clustering_coeff_distribution(G):
    """Local clustering coefficient per node."""
    G = ensure_nx(G)
    return np.array(list(nx.clustering(G).values()))


def bc_distribution(G):
    """Betweenness centrality per node."""
    G = ensure_nx(G)
    return np.array(list(nx.betweenness_centrality(G).values()))


def cc_distribution(G):
    """Closeness centrality per node."""
    G = ensure_nx(G)
    return np.array(list(nx.closeness_centrality(G).values()))


def similarity_score(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if len(a) == 0 or len(b) == 0:
        return 0.0

    dist      = wasserstein_distance(a, b)
    combined  = np.concatenate([a, b])
    val_range = combined.max() - combined.min()

    if val_range == 0:
        return 100.0   # both distributions are identical constants

    return max(0.0, (1 - dist / val_range) * 100)




   


