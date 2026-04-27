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



def make_convolution(in_channels, out_channels):
            return GINConv(nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ))

        class GINClassification(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
                super(GINClassification, self).__init__()
                self.conv1 = make_convolution(in_channels, hidden_channels)
                self.conv2 = make_convolution(hidden_channels, hidden_channels)
                self.conv3 = make_convolution(hidden_channels, out_channels)
                self.classifier = nn.Linear(out_channels, num_classes)


            def forward(self, x, edge_index, batch):
                x = self.conv1(x, edge_index)
                x = self.conv2(x, edge_index)
                x = self.conv3(x, edge_index)
                x = global_mean_pool(x, batch=batch)
                return self.classifier(x)

            def extract_embedding(self, x, edge_index, batch):
                x = self.conv1(x, edge_index)
                x = self.conv2(x, edge_index)
                x = self.conv3(x, edge_index)
                x = global_mean_pool(x, batch=batch)
                return x

        def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=1000, model_save_path='models/'):
            best_model_path = os.path.join(model_save_path, 'best_model.pth')
            best_test_accuracy = 0.0
            best_f1 = 0.0

            log_data = {'Epoch': [], 'Average Loss': [], 'Test Accuracy': [], 'F1 Score': []}

            for epoch in tqdm(range(num_epochs), desc='Training'):
                total_loss = 0
                total_samples = 0

                # Training loop
                model.train()
                for data in train_loader:
                    x, edge_index, y = data.x, data.edge_index, data.y
                    optimizer.zero_grad()
                    out = model(x, edge_index, data.batch)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * data.num_graphs
                    total_samples += data.num_graphs

                avg_loss = total_loss / total_samples

                # Test loop
                model.eval()
                correct = 0
                total = 0
                true_labels = []
                predicted_labels = []
                with torch.no_grad():
                    for data in test_loader:
                        x, edge_index, y = data.x, data.edge_index, data.y
                        out = model(x, edge_index, data.batch)
                        _, predicted = torch.max(out, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                        true_labels.extend(y.numpy())
                        predicted_labels.extend(predicted.numpy())

                test_accuracy = correct / total
                f1 = f1_score(true_labels, predicted_labels, average='weighted')
                tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.5f}, Test Accuracy: {test_accuracy:.5f}, F1 Score: {f1:.5f}')

                # Log data
                log_data['Epoch'].append(epoch + 1)
                log_data['Average Loss'].append(avg_loss)
                log_data['Test Accuracy'].append(test_accuracy)
                log_data['F1 Score'].append(f1)

                # Save the best model based on test accuracy and F1-score
                if test_accuracy > best_test_accuracy or (test_accuracy == best_test_accuracy and best_f1 < f1):
                    tqdm.write("$$$ best model is updated according to accuracy! $$$")
                    best_test_accuracy = test_accuracy
                    best_f1 = f1
                    os.makedirs(model_save_path, exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)

            # Load the best model for evaluation
            model.load_state_dict(torch.load(best_model_path))

            # Convert log_data to DataFrame
            log_df = pd.DataFrame(log_data)
            log_df.to_csv(os.path.join(model_save_path, 'train_log.csv'), index=False)