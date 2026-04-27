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




def load_nci1():
    NCI1 = TUDataset(root=f'data/NCI1', name='NCI1', use_node_attr=True)
    return

def load_polblogs():
    PolBlogs_data = PolBlogs(root="data/PolBlogs")
    Blogs = PolBlogs_data[0]
    def build_polblogs_dataset(data,
                           num_samples=1500,
                           k=2,
                           min_nodes=20,
                           max_nodes=40):

        graphs = []
        num_nodes = data.num_nodes

        while len(graphs) < num_samples:

            center = random.randint(0, num_nodes - 1)

            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                center,
                k,
                data.edge_index,
                relabel_nodes=True
            )

            n = subset.size(0)

            if n < min_nodes or n > max_nodes:
                continue

            node_labels = data.y[subset]

            # Node features = one-hot labels
            x = F.one_hot(node_labels, num_classes=2).float()

            # Graph label = majority vote
            graph_label = torch.mode(node_labels).values.unsqueeze(0)

            graph = Data(
                x=x,
                edge_index=edge_index,
                y=graph_label
            )

            graphs.append(graph)
        return graphs
    PolBlogss = build_polblogs_dataset(Blogs, num_samples=1500)
    return PolBlogss

def load_enzymes():
    ENZYMES = TUDataset(root="data/TUDataset", name="ENZYMES")
    return ENZYMES

def load_cora(dataset_name="Cora", root="data", k_hop=2):

    if dataset_name.lower() == "cora":
        dataset = Planetoid(root=root, name="Cora")
        
    else:
        raise ValueError("Unsupported dataset. Choose 'Cora' or 'USA'.")

    data = dataset[0]
    graphs = []

    num_nodes = data.num_nodes

    for node_idx in range(num_nodes):

        # ---- Extract k-hop subgraph ----
        subset, sub_edge_index, _, sub_mapping = k_hop_subgraph(
            node_idx,
            num_hops=k_hop,
            edge_index=data.edge_index,
            relabel_nodes=True
        )

        # ✅ INSERT HERE (CRITICAL)
        if sub_edge_index.size(1) == 0:
            continue   # skip empty graphs

        # ---- Node features ----
        x_sub = data.x[subset]

        # ---- Graph label ----
        node_labels = data.y[subset]
        majority_label = Counter(node_labels.tolist()).most_common(1)[0][0]
        y_graph = torch.tensor([majority_label], dtype=torch.long)

        # ---- Create Data object ----
        graphs.append(Data(x=x_sub, edge_index=sub_edge_index, y=y_graph))

    return graphs



