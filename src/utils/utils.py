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
import argparse
import config
# ── project config ─────────────────────────────────────────────────────────
from config import (DATASET_CONFIGS, ARCH_CONFIGS, GIN_CONFIG, ATTACK_CONFIGS, DEFAULT_SEEDS, DEFAULT_DATA_ROOT,
                     DEFAULT_MODEL_ROOT, DEFAULT_OUTPUT_ROOT, get_train_config, get_gen_config)


# set seed for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# set device
def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
 

def node_to_graph_dataset(dataset_name: str, root: str, k_hop: int):
    if dataset_name.lower() == "cora":
        raw = Planetoid(root=root, name="Cora")
    elif dataset_name.lower() == "usa":
        raw = Airports(root=root, name="USA")
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from {config.SUPPORTED_DATASETS}.")
 
    data = raw[0]
    graphs = []
 
    for node_idx in range(data.num_nodes):
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx,
            num_hops=k_hop,
            edge_index=data.edge_index,
            relabel_nodes=True,
        )
        if sub_edge_index.size(1) == 0:
            continue
        x_sub = data.x[subset]
        node_labels = data.y[subset]
        majority_label = Counter(node_labels.tolist()).most_common(1)[0][0]
        y_graph = torch.tensor([majority_label], dtype=torch.long)
        graphs.append(Data(x=x_sub, edge_index=sub_edge_index, y=y_graph))
 
    return graphs


class NCI1GraphDataset(Dataset):
    def __init__(self, data, classes, max_nodes):
        print("Preparing graph dataset …")
        self.datasets  = data
        self.max_nodes = max_nodes
        self.num_classes = classes
 
        self.valid_data = []
        for d in self.datasets:
            if (d.x is not None and d.x.shape[1] > 0
                    and config.MIN_NODES <= d.num_nodes <= max_nodes
                    and d.edge_index.shape[1] > 0):
                avg_degree = d.edge_index.shape[1] / d.num_nodes
                if avg_degree <= config.MAX_AVG_DEGREE:
                    self.valid_data.append(d)
 
        all_features     = torch.cat([d.x for d in self.valid_data], dim=0)
        self.feat_mean   = all_features.mean(dim=0, keepdim=True)
        self.feat_std    = all_features.std(dim=0,  keepdim=True) + 1e-8
        self.feature_dim = self.valid_data[0].x.shape[1]
 
        self.class_stats = {}
        for label in range(self.num_classes):
            cg = [d for d in self.valid_data if d.y.item() == label]
            if not cg:
                self.class_stats[label] = {"avg_nodes": 0, "avg_edges": 0, "count": 0}
                continue
            self.class_stats[label] = {
                "avg_nodes": int(np.mean([d.num_nodes for d in cg])),
                "avg_edges": int(np.mean([d.edge_index.shape[1] / 2 for d in cg])),
                "count":     len(cg),
            }
 
        print(f"  Valid graphs : {len(self.valid_data)}")
        print(f"  Feature dim  : {self.feature_dim}")
        for lbl, st in self.class_stats.items():
            print(f"  Class {lbl}: {st['count']} graphs, "
                  f"avg {st['avg_nodes']} nodes, {st['avg_edges']} edges")
 
    def __len__(self):
        return len(self.valid_data)
 
    def __getitem__(self, idx):
        d             = self.valid_data[idx]
        features_norm = (d.x - self.feat_mean) / self.feat_std
        adj           = to_dense_adj(d.edge_index, max_num_nodes=self.max_nodes)[0]
        features      = torch.zeros(self.max_nodes, self.feature_dim)
        features[:d.num_nodes] = features_norm
        mask          = torch.zeros(self.max_nodes)
        mask[:d.num_nodes] = 1
        return adj, features, d.y.item(), mask
    

def generate_graphs_for_labels(model, labels):
    graphs = []
    print(f"\nGenerating {len(labels)} graphs …")
    for i, label in enumerate(tqdm(labels)):
        adj, feat = model.sample(label)
        graphs.append({"adj": adj, "features": feat, "label": label})
        if (i + 1) % config.GC_FLUSH_EVERY == 0:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    return graphs


def list_of_dicts_to_pyg(data_list):
    pyg = []
    for d in data_list:
        edge_index = d["adj"].nonzero(as_tuple=False).t().contiguous()
        pyg.append(Data(
            x=d["features"],
            edge_index=edge_index,
            y=torch.tensor([d["label"]], dtype=torch.long)))
    return pyg


def target_dataset_to_pyg(target_dataset):
    """Wrap plain PyG Data objects so both lists share the same type."""
    out = []
    for d in target_dataset:
        out.append(Data(edge_index=d.edge_index, x=d.x.float(), y=d.y))
    return out



def summarise_results(experiment_type, prefixes):
    files = glob.glob("*.csv")
    out_path = f"{config.RESULT_DIR}/{experiment_type}_metrics_summary.txt"
    with open(out_path, "w") as fh:
        fh.write("Metric Summary (Mean ± Std)\n")
        fh.write("=" * 50 + "\n\n")
        for prefix in prefixes:
            for group_label, selector in [
                ("Distance metrics",  lambda f, p: os.path.basename(f)[:3] == p),
                ("Structural metrics", lambda f, p: os.path.basename(f)[11:14] == p),
            ]:
                matched = [f for f in files if selector(f, prefix)]
                if not matched:
                    continue
                combined = pd.concat([pd.read_csv(f) for f in matched], ignore_index=True)
                summary  = pd.DataFrame({"Mean": combined.mean(numeric_only=True),
                                          "Std":  combined.std(numeric_only=True)})
                fh.write(f"Dataset: {prefix}  |  {group_label}\n")
                fh.write(summary.to_string())
                fh.write("\n\n")
            fh.write("-" * 50 + "\n\n")
    print(f"Results saved → {out_path}")







#############################################################################################################################
#############################################################################################################################
# load dataset function 

def load_dataset(dataset_name: str, data_root: str):
    """Load and return the raw dataset object (or list for ego-graph types)."""
    dcfg = DATASET_CONFIGS[dataset_name]
    src  = dcfg["source"]

    if src == "TUDataset":
        return TUDataset(root=os.path.join(data_root, "TUDataset"),
                         name=dcfg["name"])
    elif src == "Planetoid":
        raw = Planetoid(root=data_root, name=dcfg["name"])
        return node_to_graph_dataset(raw[0], dcfg["k_hop"])
    elif src == "Airports":
        raw = Airports(root=data_root, name=dcfg["name"])
        return node_to_graph_dataset(raw[0], dcfg["k_hop"])
    elif src == "PolBlogs":
        raw = PolBlogs(root=data_root)
        return node_to_graph_dataset(raw[0], dcfg["k_hop"])
    else:
        raise ValueError(f"Unknown dataset source: {src}")

# Data preprocessing for Cora and PolBlogs
def node_to_graph_dataset(data, k_hop: int):
    """Convert a node-classification graph to a list of ego-subgraph Data objects."""
    graphs = []
    for node_idx in range(data.num_nodes):
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx, num_hops=k_hop,
            edge_index=data.edge_index, relabel_nodes=True
        )
        if sub_edge_index.size(1) == 0:
            continue
        x_sub      = data.x[subset] if data.x is not None else torch.ones(len(subset), 1)
        node_labels = data.y[subset]
        majority    = Counter(node_labels.tolist()).most_common(1)[0][0]
        graphs.append(Data(
            x=x_sub,
            edge_index=sub_edge_index,
            y=torch.tensor([majority], dtype=torch.long)
        ))
    return graphs

# Infer node features.
def infer_node_features(dataset) -> int:
    """Return the node-feature dimension for a dataset (list or TUDataset)."""
    if hasattr(dataset, "num_node_features"):
        return dataset.num_node_features
    sample = dataset[0]
    if sample.x is not None:
        return sample.x.shape[1]
    return 1

class GraphDatasetWrapper(Dataset):
    """Pad graphs to max_nodes and return (adj, features, label, mask)."""

    def __init__(self, data_list, max_nodes, num_classes, min_nodes=2, max_avg_degree=5.0):
        self.max_nodes   = max_nodes
        self.num_classes = num_classes
        self.valid_data  = []

        for d in data_list:
            if d.x is None or d.x.shape[1] == 0:
                continue
            if not (min_nodes <= d.num_nodes <= max_nodes):
                continue
            if d.edge_index.shape[1] == 0:
                continue
            avg_deg = d.edge_index.shape[1] / d.num_nodes
            if avg_deg <= max_avg_degree:
                self.valid_data.append(d)

        if not self.valid_data:
            raise RuntimeError("No valid graphs found after filtering.")

        all_feats        = torch.cat([d.x for d in self.valid_data], dim=0)
        self.feat_mean   = all_feats.mean(0, keepdim=True)
        self.feat_std    = all_feats.std(0,  keepdim=True) + 1e-8
        self.feature_dim = self.valid_data[0].x.shape[1]

        # Class statistics
        self.class_stats = {}
        for lbl in range(num_classes):
            cg = [d for d in self.valid_data if d.y.item() == lbl]
            if cg:
                self.class_stats[lbl] = {
                    "avg_nodes": int(np.mean([d.num_nodes for d in cg])),
                    "avg_edges": int(np.mean([d.edge_index.shape[1] / 2 for d in cg])),
                    "count":     len(cg),
                }

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        d             = self.valid_data[idx]
        feats_norm    = (d.x - self.feat_mean) / self.feat_std
        adj           = to_dense_adj(d.edge_index, max_num_nodes=self.max_nodes)[0]
        features      = torch.zeros(self.max_nodes, self.feature_dim)
        features[:d.num_nodes] = feats_norm
        mask          = torch.zeros(self.max_nodes)
        mask[:d.num_nodes] = 1
        return adj, features, d.y.item(), mask

def list_of_dicts_to_pyg(data_list):
    pyg = []
    for d in data_list:
        ei = d["adj"].nonzero(as_tuple=False).t().contiguous()
        pyg.append(Data(x=d["features"], edge_index=ei,
                        y=torch.tensor([d["label"]], dtype=torch.long)))
    return pyg


def dataset_to_pyg(dataset_list):
    """Convert a list of Data objects to plain PyG Data list (no-op, just standardise)."""
    return list(dataset_list)

def generate_graphs(gen_model, labels, arch, device):
    graphs = []
    for label in tqdm(labels, desc="Generating"):
        if arch == "proposed":
            adj, feat = gen_model.sample(label)
        elif arch == "random":
            adj, feat = gen_model.sample(label)
        else:   # base / CVAE
            adj, feat = gen_model.sample(label)
        graphs.append({"adj": adj, "features": feat, "label": label})
    return graphs