

import os
import gc
import glob
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx

from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance

from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, Airports
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_networkx, k_hop_subgraph

from grakel import Graph as GrakelGraph
from grakel.kernels import WeisfeilerLehman, VertexHistogram

import config
from src.metrics import metrics

from src.utils import  utils
from src.models.gnn import make_convolution, GINClassification

from train.train_target import train_gin

from src.models.diffusion import GraphDiffusionNetwork, GraphDiffusionModel
from train.train_diffusion import train_diffusion
from src.metrics.metrics import GraphEncoder, compute_fgd_scores, compute_graph_distances_list, compute_structural_scores





def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="LOGRA: Graph Generation via Laplace-noise Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Experiment ──────────────────────────────────────────────────────────
    p.add_argument("--experiment-type", default=config.EXPERIMENT_TYPE,
                   help="Tag written into output filenames (e.g. proposed, baseline).")
    p.add_argument("--summarise-only", action="store_true",
                   help="Skip training; only aggregate existing CSV results.")

    # ── Data ────────────────────────────────────────────────────────────────
    p.add_argument("--dataset", choices=config.SUPPORTED_DATASETS,
                   default="Cora",
                   help="Dataset to use.")
    p.add_argument("--data-root", default=config.DATA_ROOT,
                   help="Root directory for raw dataset downloads.")
    p.add_argument("--k-hop", type=int, default=config.K_HOP,
                   help="Number of hops for k-hop subgraph extraction.")

    # ── Seeds ────────────────────────────────────────────────────────────────
    p.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS,
                   help="Random seeds to iterate over.")

    # ── GIN Classifier ───────────────────────────────────────────────────────
    p.add_argument("--gin-epochs", type=int, default=None,
                   help="Override GIN training epochs (default from DATASET_CONFIG).")
    p.add_argument("--gin-batch-size", type=int, default=config.GIN_BATCH_SIZE,
                   help="Batch size for GIN training.")
    p.add_argument("--load-gin", default=None, metavar="PATH",
                   help="Path to a pre-trained GIN checkpoint (skips GIN training).")

    # ── Diffusion model ───────────────────────────────────────────────────────
    p.add_argument("--diff-epochs", type=int, default=None,
                   help="Override diffusion training epochs (default from DATASET_CONFIG).")
    p.add_argument("--diff-batch-size", type=int, default=config.DIFF_BATCH_SIZE,
                   help="Batch size for diffusion training.")
    p.add_argument("--diff-lr", type=float, default=config.DIFF_LR,
                   help="Learning rate for the diffusion model.")
    p.add_argument("--diff-timesteps", type=int, default=config.DIFF_NUM_TIMESTEPS,
                   help="Number of diffusion timesteps.")
    p.add_argument("--diff-epsilon", type=float, default=config.DIFF_EPSILON,
                   help="Epsilon (scale) for the Laplace noise distribution.")

    # ── Hardware ─────────────────────────────────────────────────────────────
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even when MPS/CUDA is available.")

    return p


def run_one(dataset_name: str, seed: int, args, device: torch.device):
    cfg          = config.DATASET_CONFIG[dataset_name]
    classes      = cfg["classes"]
    MAX_NODE     = cfg["MAX_NODE"]
    num_epoch1   = args.gin_epochs   if args.gin_epochs   else cfg["num_epoch1"]
    num_epoch2   = args.diff_epochs  if args.diff_epochs  else cfg["num_epoch2"]
    types        = args.experiment_type

    print(f"\n{'='*70}")
    print(f"  Dataset={dataset_name}  Seed={seed}  Type={types}")
    print(f"{'='*70}")

    # ── Build graph dataset ──────────────────────────────────────────────────
    full_dataset = utils.node_to_graph_dataset(dataset_name, args.data_root, args.k_hop)
    node_features = full_dataset[0].num_node_features

    target_dataset, shadow_dataset = train_test_split(
        full_dataset, test_size=config.SHADOW_SPLIT_RATIO,
        random_state=config.SPLIT_RANDOM_STATE)
    train_dataset, test_dataset = train_test_split(
        target_dataset, test_size=config.TEST_SPLIT_RATIO,
        random_state=config.SPLIT_RANDOM_STATE)

    train_loader = DataLoader(train_dataset, batch_size=args.gin_batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  shuffle=False)

    # ── GIN Classifier ───────────────────────────────────────────────────────
    model_save_path = os.path.join(config.MODELS_DIR, dataset_name)
    gin_ckpt        = os.path.join(model_save_path, config.BEST_MODEL_FILENAME)

    model_class = GINClassification(
        in_channels=node_features,
        hidden_channels=config.GIN_HIDDEN_CHANNELS,
        out_channels=config.GIN_OUT_CHANNELS,
        num_classes=classes)

    if args.load_gin:
        print(f"Loading pre-trained GIN from {args.load_gin}")
        model_class.load_state_dict(torch.load(args.load_gin, map_location="cpu"))
    else:
        optimizer = optim.Adam(model_class.parameters(), lr=config.GIN_LR)
        criterion = nn.CrossEntropyLoss()
        model_class = train_gin(model_class, train_loader, test_loader,
                                optimizer, criterion, num_epoch1, model_save_path)

    # ── Collect shadow-set predictions → Total_data ───────────────────────
    model_class.eval()
    Total_data = []
    for data in shadow_dataset:
        data   = data.to("cpu")
        logits = model_class(data.x, data.edge_index, None)
        pred   = F.softmax(logits, dim=1).argmax(dim=1)
        if data.y.item() == pred.item():
            Total_data.append(data.cpu())
        else:
            data.y = pred
            Total_data.append(data.cpu())

    # ── Diffusion model ───────────────────────────────────────────────────
    diff_dataset = utils.NCI1GraphDataset(Total_data, classes, max_nodes=MAX_NODE)
    diff_loader  = DataLoader(diff_dataset, batch_size=args.diff_batch_size,
                              shuffle=True, num_workers=0)

    diff_model = GraphDiffusionModel(
        max_nodes=MAX_NODE,
        feature_dim=node_features,
        num_classes=classes,
        num_timesteps=args.diff_timesteps,
        epsilon=args.diff_epsilon,
        device=device)

    train_diffusion(diff_model, diff_loader, num_epochs=num_epoch2, lr=args.diff_lr)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(diff_model.state_dict(),
               os.path.join(config.MODELS_DIR, config.BEST_MODEL_FILENAME))

    # ── Generate graphs matching target labels ─────────────────────────────
    model_class.eval()
    labels = []
    for data in target_dataset:
        data = data.to("cpu")
        out  = model_class(data.x, data.edge_index, None)
        pred = out.argmax(dim=1)
        data.y = pred
        labels.append(pred.item())

    generated_graphs = utils.generate_graphs_for_labels(diff_model, labels)
    generated_graph  = utils.list_of_dicts_to_pyg(generated_graphs)
    original_graph   = utils.target_dataset_to_pyg(target_dataset)

    # ── FGD ───────────────────────────────────────────────────────────────
    encoder  = metrics.GraphEncoder(node_features=node_features)
    fgd_list, fgd_mean, fgd_std = compute_fgd_scores(
        original_graph, generated_graph, encoder)
    print(f"FGD  mean={fgd_mean:.4f}  std={fgd_std:.4f}")

    # ── Graph distances ────────────────────────────────────────────────────
    euclid_list, mmd_list, kernel_list, spectral_list = \
        compute_graph_distances_list(original_graph, generated_graph, node_features)

    distance_df = pd.DataFrame({
        "fgd_mean":       fgd_list,
        "euclidean_mean": euclid_list,
        "mmd_mean":       mmd_list,
        "kernel_mean":    kernel_list,
        "spectral_mean":  spectral_list,
    })
    dist_file = f"{dataset_name}_{types}_seed{seed}.csv"
    distance_df.to_csv(dist_file, index=False)
    print(f"Saved → {dist_file}")

    # ── Structural scores ──────────────────────────────────────────────────
    deg_s, clust_s, bc_s, cc_s = compute_structural_scores(
        original_graph, generated_graph)

    structural_df = pd.DataFrame({
        "degree_scores":     deg_s,
        "clustering_scores": clust_s,
        "bc_scores":         bc_s,
        "cc_scores":         cc_s,
    })
    struct_file = f"structural_{dataset_name}_{types}_seed{seed}.csv"
    structural_df.to_csv(struct_file, index=False)
    print(f"Saved → {struct_file}")


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.summarise_only:
        utils.summarise_results(args.experiment_type, config.DATASET_PREFIXES)
        return

    device = utils.get_device(force_cpu=args.cpu)
    print(f"Using device: {device}")

    for seed in args.seeds:
        utils.set_seed(seed)
        run_one(args.dataset, seed, args, device)

    utils.summarise_results(args.experiment_type, config.DATASET_PREFIXES)


if __name__ == "__main__":
    main()