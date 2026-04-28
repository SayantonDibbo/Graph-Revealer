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









generated_graph = list_of_dicts_to_pyg(generated_graphs)
original_graph = convert_generated_to_pyg(target_dataset)












# Distributional metrics----------------------------//////////////----------------------------------------////////////////
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

  


# STRUCTURAL METRICS----------------------------//////////////----------------------------------------////////////////
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