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

def split_data(dataset):
    target_dataset, shadow_dataset = train_test_split(dataset, test_size=0.5, random_state=42)

    train_dataset, test_dataset = train_test_split(target_dataset, test_size=0.2, random_state=42)

    return train_dataset, test_dataset, shadow_dataset, target_dataset


def draw_random_graph_samples(dataset, dataset_name, num_classes, num_samples=20, model_save_path='models/'):
    num_rows = 4
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
    fig.suptitle('Random Samples of Graphs')

    # Define a colormap based on the number of unique classes
    cmap = plt.cm.get_cmap('tab10', num_classes)

    # Get random indices for sampling
    random_indices = random.sample(range(len(dataset)), num_samples)

    for i, index in enumerate(random_indices):
        row = i // num_cols
        col = i % num_cols

        data = dataset[index]
        edge_index, y = data.edge_index, data.y
        G = nx.Graph()
        for src, dst in edge_index.t().tolist():
            G.add_edge(src, dst)

        # Map class labels to colors
        nodes_color = cmap(y[0])

        pos = nx.spring_layout(G)  # Positions for all nodes
        nx.draw(G, pos, with_labels=True, node_color=nodes_color, node_size=100, edge_color='k', linewidths=1, font_size=5, ax=axs[row, col])
        axs[row, col].set_title(f'Graph Sample {index}')

    # Custom legend for class labels
    handles = [plt.Line2D([], [], color=cmap(i), marker='o', linestyle='', markersize=10, label=f'Class {i}') for i in range(num_classes)]
    plt.legend(handles=handles, title='Class Labels', loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the plot
    plt.savefig(os.path.join(model_save_path, f'{dataset_name}/random_samples_of_graphs.png'))



    def extract_embeddings(model, dataset):
        model.eval()
        embeddings_list = []
        for data in dataset:
            x, edge_index = data.x, data.edge_index
            with torch.no_grad():
                embedding = model.extract_embedding(x, edge_index, data.batch)
            embeddings_list.append(embedding.numpy())
        embeddings_all = np.concatenate(embeddings_list)
        return embeddings_all

    def predict_labels(model, test_loader):
        predicted_labels = []
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                x, edge_index, _ = data.x, data.edge_index, data.batch
                out = model(x, edge_index, data.batch)
                _, predicted = torch.max(out, 1)
                predicted_labels.extend(predicted.numpy())
        return predicted_labels

    def collect_labels(loader):
        # Get true labels for test set
        true_labels = []
        for data in loader:
            true_labels.append(data.y.numpy())
        true_labels = np.concatenate(true_labels)
        return true_labels

    def visualize_embeddings(embeddings, true_labels, predicted_labels, model_save_path):
        tsne = TSNE(n_components=2)
        embeddings_tsne = tsne.fit_transform(embeddings)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('t-SNE Visualization of Embeddings')

        for ax, labels, title in zip(axs, [true_labels, predicted_labels], ['True Labels', 'Predicted Labels']):
            for label in np.unique(labels):
                indices = np.where(labels == label)
                ax.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1], label=f'Class {label}')
            ax.set_title(title)
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.legend()
            ax.grid(True)

        plt.savefig(os.path.join(model_save_path, 'test_embedding_with_labels.png'))

def generate_graphs_for_labels(models, labels):
            """Generate graphs for list of labels"""
            graphs = []
            print(f"\nGenerating {len(labels)} graphs...")
        
            for i, label in enumerate(tqdm(labels)):
                adj, feat = gan.sample(label)
                graphs.append({'adj': adj, 'features': feat, 'label': label})
        
                if (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
        
            return graphs

def convert_generated_to_pyg(generated_graph):
            pyg_data = []
            for graph in generated_graph:
                edge_index = graph['edge_index'].long()
                x = graph['x'].float()

                # Convert condition to graph label
                condition = graph.get('y', None)
                if condition is not None:
                    y = torch.tensor([condition], dtype=torch.long)
                else:
                    y = None

                data = Data(
                    edge_index=edge_index,
                    x=x,
                    y=y
                )
                pyg_data.append(data)

            return pyg_data

def list_of_dicts_to_pyg(data_list):
            pyg_graphs = []
            for data_dict in data_list:
                adj = data_dict['adj']
                x = data_dict['features']
                y = torch.tensor([data_dict['label']], dtype=torch.long)

                # Convert adjacency matrix to edge_index
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()

                pyg_graph = Data(x=x, edge_index=edge_index, y=y)
                pyg_graphs.append(pyg_graph)

            return pyg_graphs

def visualize_graphs(original_graphs, generated_graphs, dataset_name, seed_value):
            """
            Visualize original and generated graphs in 2 rows and multiple columns,
            and save the figure to a PDF named {dataset_name}_seed{seed_value}_graphs.pdf
            """
            num_graphs = len(original_graphs)

            if num_graphs != len(generated_graphs):
                raise ValueError("Original and generated graph lists must be the same length.")

            # Create 2 rows, num_graphs columns
            fig, axes = plt.subplots(2, num_graphs, figsize=(4*num_graphs, 8))
            fig.suptitle(f"{dataset_name} - Seed {seed_value} Graph Comparison", fontsize=16)

            for i in range(num_graphs):
                # Convert to NetworkX
                G_orig = to_networkx(original_graphs[i], to_undirected=True)
                G_gen = to_networkx(generated_graphs[i], to_undirected=True)

                # Plot original (row 0)
                ax = axes[0, i] if num_graphs > 1 else axes[0]
                nx.draw(G_orig, ax=ax, node_size=50, with_labels=False)
                ax.set_title(f"Original {i+1}")

                # Plot generated (row 1)
                ax = axes[1, i] if num_graphs > 1 else axes[1]
                nx.draw(G_gen, ax=ax, node_size=50, with_labels=False)
                ax.set_title(f"Generated {i+1}")

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Save PDF dynamically
            filename = f"{dataset_name}_seed{seed_value}_graphs.pdf"
            plt.savefig(filename)
            plt.close()
            print(f"Saved {filename}")

        