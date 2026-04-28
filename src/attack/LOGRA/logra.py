# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import PolBlogs
from torch_geometric.utils import k_hop_subgraph
from collections import Counter    
from torch_geometric.data import DataLoader 


# Function to perform logra attack on target model
def logra(model_class, shadow_dataset,device = "cpu"):
    correct_graphs = []
    incorrect_graphs = []
    Total_data = []
    model_class.eval()
    for data in shadow_dataset:
        data = data.to(device)

        out = model_class(data.x, data.edge_index, None)
        pred = out.argmax(dim=1)

        if pred.item() == data.y.item():      # correct prediction
            correct_graphs.append(data.cpu())
            Total_data.append(data)
        else:
            incorrect_graphs.append(data.cpu())

    correct_loader = DataLoader(correct_graphs, batch_size=32)
    incorrect_loader = DataLoader(incorrect_graphs, batch_size=32)

    print(len(Total_data), len(correct_graphs), len(incorrect_graphs))
    return Total_data

