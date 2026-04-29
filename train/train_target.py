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

from src.models.gnn import GINClassification
## need model here and parameters
import config

def train_classifier(model, train_loader, test_loader,
                     num_epochs, save_path):
    os.makedirs(save_path, exist_ok=True)
    best_path = os.path.join(save_path, "best_model.pth")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_f1 = 0.0, 0.0
    log = {"Epoch": [], "Average Loss": [], "Test Accuracy": [], "F1 Score": []}

    for epoch in tqdm(range(num_epochs), desc="Classifier training"):
        model.train()
        total_loss, total_n = 0.0, 0
        for data in train_loader:
            optimizer.zero_grad()
            out  = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            total_n    += data.num_graphs

        avg_loss = total_loss / total_n

        model.eval()
        correct, total = 0, 0
        true_l, pred_l = [], []
        with torch.no_grad():
            for data in test_loader:
                out     = model(data.x, data.edge_index, data.batch)
                _, pred = torch.max(out, 1)
                total   += data.y.size(0)
                correct += (pred == data.y).sum().item()
                true_l.extend(data.y.numpy())
                pred_l.extend(pred.numpy())

        acc = correct / total
        f1  = f1_score(true_l, pred_l, average="weighted")

        log["Epoch"].append(epoch + 1)
        log["Average Loss"].append(avg_loss)
        log["Test Accuracy"].append(acc)
        log["F1 Score"].append(f1)

        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc, best_f1 = acc, f1
            torch.save(model.state_dict(), best_path)
            tqdm.write(f"  [epoch {epoch+1}] New best  acc={acc:.4f}  f1={f1:.4f}")

    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    pd.DataFrame(log).to_csv(os.path.join(save_path, "train_log.csv"), index=False)
    return model