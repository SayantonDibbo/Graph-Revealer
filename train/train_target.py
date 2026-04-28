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

from src.models import gnn
from utils import utils




def main():
    batch_size = 64
    num_epochs = num_epoch

    dataset = utils.load_nci1()

    target_dataset, shadow_dataset = train_test_split(dataset, test_size=0.5, random_state=42)

    # Split dataset into training and test sets
    train_dataset, test_dataset = train_test_split(target_dataset, test_size=0.2, random_state=42)

    # Draw graphs from the test dataset before training
    utils.draw_random_graph_samples(test_dataset, dataset_name, 2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    model = gnn.GINClassification(in_channels=dataset[0].num_node_features, hidden_channels=1000, out_channels=100, num_classes=2)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    gnn.train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=num_epochs, model_save_path=f'models/{dataset_name}')

    # Extract embeddings for test set
    test_embeddings = utils.extract_embeddings(model, test_dataset)

    # Collect true labels
    true_labels = utils.collect_labels(test_loader)

    # Predict labels for test set
    predicted_labels = utils.predict_labels(model, test_loader)

    # Plot t-SNE visualization with true and predicted labels
    utils.visualize_embeddings(test_embeddings, true_labels, predicted_labels, model_save_path=f'models/{dataset_name}')
    return model

if __name__ == "__main__":
    model_class = main()
