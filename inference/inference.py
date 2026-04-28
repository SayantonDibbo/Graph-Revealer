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

from src.utils.utils import generate_graphs_for_labels
 ## need model_class, model here and dataloader.

def generate_data(model_class, model target_dataset, device='cpu'):
    labels = []
    model_class.eval()
    for data in target_dataset:
        data = data.to(device)
        
        out = model_class(data.x, data.edge_index, None)
        pred = out.argmax(dim=1)

        if pred.item() == data.y.item():      # correct prediction
            
            labels.append(pred.item())
        else:
            data.y = pred

            labels.append(pred.item())

    #correct_loader = DataLoader(correct_graphs, batch_size=32)
    #incorrect_loader = DataLoader(incorrect_graphs, batch_size=32)
    print("Total labels predicted:",len(labels))


    generated_graphs = generate_graphs_for_labels(model, labels)
    return generated_graphs