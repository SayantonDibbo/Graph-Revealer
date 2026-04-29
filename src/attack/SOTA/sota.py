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
import copy

device = "cpu"
def sota_attack(classifier, shadow_ds, device='cpu'):
    # Step 1: Generate data for attack model
    correct_graphs   = []
    incorrect_graphs = []
    total_data       = []

    for data in shadow_ds:
        data = data.to("cpu")
        with torch.no_grad():
            out  = classifier(data.x, data.edge_index, None)
            pred = out.argmax(dim=1)

        if pred.item() == data.y.item():        # correct prediction only
            cloned   = copy.deepcopy(data)
            cloned.y = torch.tensor([pred.item()], dtype=torch.long)
            correct_graphs.append(cloned)
            total_data.append(cloned)
        else:
            cloned   = copy.deepcopy(data)
            cloned.y = torch.tensor([pred.item()], dtype=torch.long)
            total_data.append(cloned)
            incorrect_graphs.append(data.cpu())

    correct_loader   = DataLoader(correct_graphs,   batch_size=32)
    incorrect_loader = DataLoader(incorrect_graphs, batch_size=32)
    print(f"  Shadow dataset — Total: {len(shadow_ds)}  "
            f"Correct: {len(correct_graphs)}  Incorrect: {len(incorrect_graphs)}")
    return total_data