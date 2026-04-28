
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




# -------------------------------
# Convert PyG Data → GraKeL Graph
# -------------------------------
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


# -------------------------------
# Euclidean Distance (embeddings)
# -------------------------------
emb1 = original_graph
emb2 = generated_graph
def euclidean_distance(emb1, emb2):
    return float(torch.norm(emb1 - emb2, p=2))


# -------------------------------
# Maximum Mean Discrepancy (MMD)
# -------------------------------
def mmd_distance(emb1, emb2, sigma=1.0):
    def gaussian_kernel(x, y, sig):
        return torch.exp(-torch.norm(x - y, p=2)**2 / (2 * sig**2))

    return float(
        gaussian_kernel(emb1, emb1, sigma)
        + gaussian_kernel(emb2, emb2, sigma)
        - 2 * gaussian_kernel(emb1, emb2, sigma)
    )


# -------------------------------
# Graph Kernel Similarity (WL)
# -------------------------------
def graph_kernel_similarity(original_graph, generated_graph, h=2):
    g1 = pyg_to_grakel_graph(original_graph)
    g2 = pyg_to_grakel_graph(generated_graph)

    wl_kernel = WeisfeilerLehman(n_iter=h, base_graph_kernel=VertexHistogram)
    K = wl_kernel.fit_transform([g1, g2])

    return float(K[0, 1])


# -------------------------------
# Spectral Distance (Laplacian)
# -------------------------------
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


# -------------------------------
# Compute metrics for all pairs
# -------------------------------
def compute_graph_distances_list(
    original_graph,
    generated_graph,
    encoder,
    device="cpu"
):
    assert len(original_graph) == len(generated_graph), \
        "Lists must be equal length."
    encoder = GraphEncoder(node_features=node_features)
    encoder = encoder.to(device)
    encoder.eval()

    # Lists for each metric
    euclid_list = []
    mmd_list = []
    kernel_list = []
    spectral_list = []

    # List of dictionaries (per graph pair)
    metrics_per_graph = []

    for o_graph, g_graph in zip(original_graph, generated_graph):

        o_graph = o_graph.to(device)
        g_graph = g_graph.to(device)

        # batch vectors
        o_batch = torch.zeros(o_graph.x.size(0), dtype=torch.long, device=device)
        g_batch = torch.zeros(g_graph.x.size(0), dtype=torch.long, device=device)

        # Embeddings
        with torch.no_grad():
            emb_o = encoder(o_graph.x, o_graph.edge_index, o_batch).squeeze()
            emb_g = encoder(g_graph.x, g_graph.edge_index, g_batch).squeeze()

        # Metrics
        euclid = euclidean_distance(emb_o, emb_g)
        mmd = mmd_distance(emb_o, emb_g)
        kernel = graph_kernel_similarity(o_graph, g_graph)
        spectral = spectral_distance(o_graph, g_graph)

        # Store
        euclid_list.append(euclid)
        mmd_list.append(mmd)
        kernel_list.append(kernel)
        spectral_list.append(spectral)

        metrics_per_graph.append({
            "euclidean_distance": euclid,
            "mmd": mmd,
            "graph_kernel_similarity": kernel,
            "spectral_distance": spectral
        })

    # Summary statistics
    summary = {
        "euclidean_mean": float(np.mean(euclid_list)),
        "euclidean_std": float(np.std(euclid_list)),

        "mmd_mean": float(np.mean(mmd_list)),
        "mmd_std": float(np.std(mmd_list)),

        "kernel_mean": float(np.mean(kernel_list)),
        "kernel_std": float(np.std(kernel_list)),

        "spectral_mean": float(np.mean(spectral_list)),
        "spectral_std": float(np.std(spectral_list)),
    }

    return metrics_per_graph, euclid_list, mmd_list, kernel_list, spectral_list, summary

encoder = GraphEncoder(node_features=node_features)

metrics_per_graph, euclid_list, mmd_list, kernel_list, spectral_list, summary = \
    compute_graph_distances_list(
        original_graph,
        generated_graph,
        encoder,
        device="cpu"
    )

#print(metrics_per_graph)   # list of dictionaries (one per graph)
summary = {
"fgd_mean": float(np.mean(fgd_list)),
"euclidean_mean": float(np.mean(euclid_list)),
"mmd_mean": float(np.mean(mmd_list)),
"kernel_mean": float(np.mean(kernel_list)),
"spectral_mean": float(np.mean(spectral_list)),
}

print(summary)             # mean + std for all metrics

summary = {
"fgd_mean": fgd_list,
"euclidean_mean": euclid_list,
"mmd_mean": mmd_list,
"kernel_mean": kernel_list,
"spectral_mean": spectral_list,
}
    

# Convert to DataFrame
df = pd.DataFrame(summary)

# Create filename with dataset name and seed value
filename = f"{dataset_name}_{types}_seed{seed}.csv"

# Save to CSV
df.to_csv(filename, index=False)

print(f"CSV saved as {filename}")

    
    
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


# ── Main evaluation loop ──────────────────────────────────────────────────────
degree_scores      = []
clustering_scores  = []   
bc_scores          = []
cc_scores          = []

for G_orig, G_gen in zip(original_graph, generated_graph):
    G_orig = ensure_nx(G_orig)
    G_gen  = ensure_nx(G_gen)

    degree_scores.append(similarity_score(
        degree_distribution(G_orig),
        degree_distribution(G_gen)
    ))
    clustering_scores.append(similarity_score(
        clustering_coeff_distribution(G_orig),
        clustering_coeff_distribution(G_gen)
    ))
    bc_scores.append(similarity_score(
        bc_distribution(G_orig),
        bc_distribution(G_gen)
    ))
    cc_scores.append(similarity_score(
        cc_distribution(G_orig),
        cc_distribution(G_gen)
    ))

structural = {
    "degree_scores":     degree_scores,
    "clustering_scores": clustering_scores,   # renamed from lcc_scores
    "bc_scores":         bc_scores,
    "cc_scores":         cc_scores,
}
df = pd.DataFrame(structural)

# Create filename with dataset name and seed value
filename = f"structural_{dataset_name}_{types}_seed{seed}.csv"

# Save to CSV
df.to_csv(filename, index=False)

print(f"CSV saved as {filename}")
    
prefixes = ["Cor"]
files = glob.glob("*.csv")

with open(f'{types}_metrics_summary.txt', "w") as out_file:

out_file.write("Metric Summary (Mean and Std)\n")
out_file.write("="*50 + "\n\n")

for prefix in prefixes:

    # --- First metric group (position 0:4) ---
    files_group1 = [
        file for file in files if os.path.basename(file)[:3] == prefix
    ]

    if len(files_group1) > 0:
        dfs1 = [pd.read_csv(file) for file in files_group1]
        combined1 = pd.concat(dfs1, ignore_index=True)

        mean1 = combined1.mean(numeric_only=True)
        std1 = combined1.std(numeric_only=True)

        summary1 = pd.DataFrame({"Mean": mean1, "Std": std1})

        out_file.write(f"Dataset: {prefix} (Metric Set 1)\n")
        out_file.write(summary1.to_string())
        out_file.write("\n\n")

    # --- Second metric group (position 12:16) ---
    files_group2 = [
        file for file in files if os.path.basename(file)[11:14] == prefix
    ]

    if len(files_group2) > 0:
        dfs2 = [pd.read_csv(file) for file in files_group2]
        combined2 = pd.concat(dfs2, ignore_index=True)

        mean2 = combined2.mean(numeric_only=True)
        std2 = combined2.std(numeric_only=True)

        summary2 = pd.DataFrame({"Mean": mean2, "Std": std2})

        out_file.write(f"Dataset: {prefix} (Metric Set 2)\n")
        out_file.write(summary2.to_string())
        out_file.write("\n\n")

    out_file.write("-"*50 + "\n\n")

print("Results saved to metrics_summary.txt")

