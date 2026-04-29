#!/usr/bin/env python3
# main.py
"""
LOGRA – CLI entry-point
=======================
Run any of the three graph-generation architectures
(proposed / random / base) on any supported dataset via argparse.

Examples
--------
# Proposed (Laplace diffusion) on ENZYMES, seeds 11 & 22
python main.py --arch proposed --dataset ENZYMES --seeds 11 22

# GAN (random) on NCI1 with custom epochs
python main.py --arch random --dataset NCI1 --epochs-classifier 300 --epochs-gen 70

# CVAE (base) on Cora, all default seeds
python main.py --arch base --dataset Cora

# Multiple datasets in one run
python main.py --arch proposed --dataset Cora USA

# Override generative-model LR and batch-size
python main.py --arch proposed --dataset Cora --gen-lr 5e-4 --gen-batch-size 16

# Skip classifier training (load from disk) and only run generation
python main.py --arch base --dataset ENZYMES --skip-classifier

# Dry-run: print resolved config and exit
python main.py --arch proposed --dataset Cora --dry-run
"""

import argparse
import os
import sys
import random
import glob
import copy
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import matplotlib

from src.attack import LOGRA

from src import attack
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid, Airports, PolBlogs
from torch_geometric.utils import (
    to_dense_adj, to_networkx, k_hop_subgraph
)
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

from grakel import Graph as GrakelGraph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from src.attack.LOGRA import logra
from src.attack.CSGRA import csgra
from src.attack.SOTA import sota
from src.utils.utils import load_dataset, infer_node_features, dataset_to_pyg, list_of_dicts_to_pyg
from src.models.gnn import GINClassification, make_gin_conv
from src.models.gan import GANGenerator, GANDiscriminator, GraphGAN
from src.models.autoencoder import GraphCVAE
from src.models.diffusion import GraphDiffusionModel
from src.metrics.metrics import compute_fgd, compute_metrics, compute_structural
from train.train_target import train_classifier
from src.utils.utils import GraphDatasetWrapper
from inference.inference import generate_graphs
from src.metrics.metrics import GCNEncoder
from train.train_autoencoder import train_cvae
from train.train_gan import train_gan   
from train.train_diffusion import train_diffusion



# ── project config ─────────────────────────────────────────────────────────
from config import (DATASET_CONFIGS, ARCH_CONFIGS, GIN_CONFIG, ATTACK_CONFIGS, DEFAULT_SEEDS, DEFAULT_DATA_ROOT,
                     DEFAULT_MODEL_ROOT, DEFAULT_OUTPUT_ROOT, get_train_config, get_gen_config)



# ===========================================================================
# 1.  CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="LOGRA – graph generation benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── what to run ─────────────────────────────────────────────────────────
    p.add_argument(
        "--arch", required=True,
        choices=list(ARCH_CONFIGS.keys()),
        help="Generative architecture: proposed (diffusion) | random (GAN) | base (CVAE)",
    )
    p.add_argument(
        "--dataset", required=True, nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        help="One or more datasets to run.",
    )

    # ── seeds ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        metavar="SEED",
        help="Random seeds (space-separated). Default: 11 22 33 44 55",
    )
    # ── attack type ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--attack", required=True,
        choices=list(ATTACK_CONFIGS.keys()),
        help="Attack type: logra | csgra | sota",
    )

    # ── epoch overrides ──────────────────────────────────────────────────────
    p.add_argument(
        "--epochs-classifier", type=int, default=None,
        metavar="N",
        help="Override number of classifier training epochs.",
    )
    p.add_argument(
        "--epochs-gen", type=int, default=None,
        metavar="N",
        help="Override number of generative-model training epochs.",
    )

    # ── generative-model hyperparameters ─────────────────────────────────────
    p.add_argument("--gen-lr",         type=float, default=None, metavar="LR",
                   help="Learning rate for the generative model.")
    p.add_argument("--gen-batch-size", type=int,   default=None, metavar="B",
                   help="Batch size for generative model training.")
    p.add_argument("--latent-dim",     type=int,   default=None, metavar="D",
                   help="Latent dimensionality (GAN / CVAE).")
    p.add_argument("--hidden-dim",     type=int,   default=None, metavar="D",
                   help="Hidden dimension for generator / encoder / decoder.")

    # ── CVAE-specific ─────────────────────────────────────────────────────────
    p.add_argument(
        "--beta-schedule",
        choices=["constant", "linear", "cyclical"],
        default=None,
        help="(CVAE only) Beta annealing schedule for the KL term.",
    )

    # ── GAN-specific ──────────────────────────────────────────────────────────
    p.add_argument("--lr-g", type=float, default=None,
                   help="(GAN only) Generator learning rate.")
    p.add_argument("--lr-d", type=float, default=None,
                   help="(GAN only) Discriminator learning rate.")

    # ── paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--data-root",   default=DEFAULT_DATA_ROOT,
                   help="Root folder for downloaded datasets.")
    p.add_argument("--model-root",  default=DEFAULT_MODEL_ROOT,
                   help="Root folder for saved model checkpoints.")
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                   help="Root folder for CSV / PDF outputs.")

    # ── flags ─────────────────────────────────────────────────────────────────
    p.add_argument("--skip-classifier", action="store_true",
                   help="Load the classifier from disk instead of training it.")
    p.add_argument("--skip-gen",        action="store_true",
                   help="Load the generative model from disk instead of training.")
    p.add_argument("--dry-run",         action="store_true",
                   help="Print the resolved configuration and exit without running.")
    p.add_argument("--device",          default=None,
                   help="Torch device string, e.g. cuda / cpu / mps. Auto-detected by default.")

    return p


def resolve_config(args) -> dict:
    """
    Merge CLI arguments with config.py defaults and return a single
    flat dict that the runner will use for each (arch, dataset, seed) triple.
    """
    cfg = {}

    cfg["arch"]    = args.arch
    cfg["datasets"] = args.dataset
    cfg["seeds"]   = args.seeds
    cfg["attack"]  = args.attack

    # Device
    if args.device:
        cfg["device"] = args.device
    elif torch.cuda.is_available():
        cfg["device"] = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cfg["device"] = "mps"
    else:
        cfg["device"] = "cpu"

    # Paths
    cfg["data_root"]   = args.data_root
    cfg["model_root"]  = args.model_root
    cfg["output_root"] = args.output_root

    # Flags
    cfg["skip_classifier"] = args.skip_classifier
    cfg["skip_gen"]        = args.skip_gen

    # GIN classifier
    cfg["gin"] = dict(GIN_CONFIG)

    # Per-dataset train epochs (filled per dataset in the run loop)
    cfg["epochs_classifier_override"] = args.epochs_classifier
    cfg["epochs_gen_override"]        = args.epochs_gen

    # Generative model base config
    gen_cfg = get_gen_config(args.arch)
    if args.gen_lr is not None:
        # handles both lr (diffusion/cvae) and lr_g/lr_d (GAN)
        gen_cfg["lr"]   = args.gen_lr
        gen_cfg["lr_g"] = args.gen_lr
        gen_cfg["lr_d"] = args.gen_lr
    if args.gen_batch_size is not None:
        gen_cfg["batch_size"] = args.gen_batch_size
    if args.latent_dim is not None:
        gen_cfg["latent_dim"] = args.latent_dim
    if args.hidden_dim is not None:
        gen_cfg["hidden_dim"] = args.hidden_dim
    if args.beta_schedule is not None and "beta_schedule" in gen_cfg:
        gen_cfg["beta_schedule"] = args.beta_schedule
    if args.lr_g is not None and "lr_g" in gen_cfg:
        gen_cfg["lr_g"] = args.lr_g
    if args.lr_d is not None and "lr_d" in gen_cfg:
        gen_cfg["lr_d"] = args.lr_d
    cfg["gen"] = gen_cfg

    return cfg




def run_experiment(cfg, dataset_name, seed):
    print(f"\n{'='*60}")
    print(f"  arch={cfg['arch']}  dataset={dataset_name}  seed={seed}")
    print(f"{'='*60}")

    device = cfg["device"]
    arch   = cfg["arch"]
    dcfg   = DATASET_CONFIGS[dataset_name]
    attack   = cfg["attack"]

    # ── 1. Seeding ──────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)

    # ── 2. Load dataset ──────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, cfg["data_root"])

    node_features = (
        dcfg["node_features"]
        if dcfg["node_features"] != -1
        else infer_node_features(dataset)
    )
    num_classes = dcfg["classes"]
    max_nodes   = dcfg["max_nodes"]

    print(f"  node_features={node_features}  num_classes={num_classes}  max_nodes={max_nodes}")

    # ── 3. Split ─────────────────────────────────────────────────────────────
    target_ds, shadow_ds = train_test_split(dataset, test_size=0.5, random_state=42)
    train_ds, test_ds    = train_test_split(target_ds, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_ds, batch_size=cfg["gin"]["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["gin"]["batch_size"], shuffle=False)

    # ── 4. Classifier ────────────────────────────────────────────────────────
    model_dir = os.path.join(cfg["model_root"], dataset_name, arch, str(seed))
    gin_path  = os.path.join(model_dir, "best_model.pth")

    classifier = GINClassification(
        in_channels=node_features,
        hidden_channels=cfg["gin"]["hidden_channels"],
        out_channels=cfg["gin"]["out_channels"],
        num_classes=num_classes,
    )

    tc = get_train_config(arch, dataset_name)
    n_cls = cfg["epochs_classifier_override"] or tc["num_epoch_classifier"]
    n_gen = cfg["epochs_gen_override"]        or tc["num_epoch_gen"]

    if cfg["skip_classifier"] and os.path.exists(gin_path):
        print(f"  Loading classifier from {gin_path}")
        classifier.load_state_dict(torch.load(gin_path, map_location="cpu"))
    else:
        print(f"  Training classifier for {n_cls} epochs...")
        classifier = train_classifier(classifier, train_loader, test_loader, n_cls, model_dir)

    classifier.eval()

    # ── 5. Query shadow_ds through classifier to build generative training set
    #       Only graphs where the classifier prediction matches the true label
    #       (correct predictions) are kept for generative model training.
    # ─────────────────────────────────────────────────────────────────────────
    if cfg["attack"] == "logra":
        total_data = logra.logra_attack(classifier, shadow_ds, device)
    elif cfg["attack"] == "csgra":
        total_data = csgra.csgra_attack(classifier, shadow_ds, device)
    elif cfg["attack"] == "sota":
        total_data = sota.sota_attack(classifier, shadow_ds, device)

    # ── 6. Wrap shadow correct-prediction graphs for generative training ──────
    gen_cfg        = cfg["gen"]
    batch_size_gen = gen_cfg.get("batch_size", 32)
    gen_dataset    = GraphDatasetWrapper(total_data, max_nodes, num_classes)
    gen_loader     = DataLoader(gen_dataset, batch_size=batch_size_gen, shuffle=True, num_workers=0)

    gen_model_path = os.path.join(model_dir, "gen_model.pth")

    # ── 7. Train / load generative model ─────────────────────────────────────
    if arch == "proposed":
        gen_model = GraphDiffusionModel(
            max_nodes=max_nodes,
            feature_dim=gen_dataset.feature_dim,
            num_classes=num_classes,
            num_timesteps=gen_cfg.get("num_timesteps", 100),
            epsilon=gen_cfg.get("epsilon", 0.5),
            hidden_dim=gen_cfg.get("hidden_dim", 256),
            device=device,
        )
        if cfg["skip_gen"] and os.path.exists(gen_model_path):
            gen_model.load_state_dict(torch.load(gen_model_path, map_location=device))
        else:
            train_diffusion(gen_model, gen_loader, n_gen, gen_cfg.get("lr", 1e-3))
            torch.save(gen_model.state_dict(), gen_model_path)

    elif arch == "random":
        gen_model = GraphGAN(
            latent_dim=gen_cfg.get("latent_dim", 128),
            max_nodes=max_nodes,
            feature_dim=gen_dataset.feature_dim,
            num_classes=num_classes,
            hidden_dim=gen_cfg.get("hidden_dim", 256),
            device=device,
        )
        if cfg["skip_gen"] and os.path.exists(gen_model_path):
            state = torch.load(gen_model_path, map_location=device)
            gen_model.generator.load_state_dict(state["generator"])
            gen_model.discriminator.load_state_dict(state["discriminator"])
        else:
            train_gan(gen_model, gen_loader, n_gen,
                      gen_cfg.get("lr_g", 2e-4), gen_cfg.get("lr_d", 2e-4))
            torch.save({"generator":    gen_model.generator.state_dict(),
                        "discriminator": gen_model.discriminator.state_dict()},
                       gen_model_path)

    else:  # base / CVAE
        gen_model = GraphCVAE(
            max_nodes=max_nodes,
            feature_dim=gen_dataset.feature_dim,
            num_classes=num_classes,
            hidden_dim=gen_cfg.get("hidden_dim", 256),
            latent_dim=gen_cfg.get("latent_dim", 128),
            device=device,
        )
        if cfg["skip_gen"] and os.path.exists(gen_model_path):
            gen_model.load_state_dict(torch.load(gen_model_path, map_location=device))
        else:
            train_cvae(gen_model, gen_loader, n_gen,
                       gen_cfg.get("lr", 2e-3),
                       gen_cfg.get("beta_schedule", "linear"))
            torch.save(gen_model.state_dict(), gen_model_path)

    # ── 8. Query target_ds through classifier to get labels for generation ────
    #       All target graphs are kept; mispredicted labels are overwritten with
    #       the classifier's prediction so generation is guided by the model's
    #       view of the target data.
    # ─────────────────────────────────────────────────────────────────────────
    print("  Building target graph list with predicted labels for generation...")
    gen_labels  = []
    target_data = []

    for data in target_ds:
        data = data.to("cpu")
        with torch.no_grad():
            pred = classifier(data.x, data.edge_index, None).argmax(dim=1).item()
        cloned = copy.deepcopy(data)
        if pred != data.y.item():
            cloned.y = torch.tensor([pred], dtype=torch.long)
        target_data.append(cloned)
        gen_labels.append(pred)

    # ── 9. Generate graphs using target labels ────────────────────────────────
    print("  Generating graphs...")
    generated_dicts = generate_graphs(gen_model, gen_labels, arch, device)
    generated_graph = list_of_dicts_to_pyg(generated_dicts)
    original_graph  = dataset_to_pyg(target_data)

    # ── 10. Evaluation ────────────────────────────────────────────────────────
    print("  Computing evaluation metrics...")
    encoder   = GCNEncoder(node_features)
    fgd_list  = compute_fgd(original_graph, generated_graph, encoder)
    euclid_l, mmd_l, kernel_l, spectral_l = compute_metrics(
        original_graph, generated_graph, node_features
    )
    deg_s, lcc_s, bc_s, cc_s = compute_structural(original_graph, generated_graph)

    # ── 11. Save CSVs ─────────────────────────────────────────────────────────
    out_dir = cfg["output_root"]
    os.makedirs(out_dir, exist_ok=True)

    metric_df = pd.DataFrame({
        "fgd_mean":        fgd_list,
        "euclidean_mean":  euclid_l,
        "mmd_mean":        mmd_l,
        "kernel_mean":     kernel_l,
        "spectral_mean":   spectral_l,
    })
    metric_file = os.path.join(out_dir, f"{dataset_name}_{arch}_seed{seed}.csv")
    metric_df.to_csv(metric_file, index=False)

    struct_df = pd.DataFrame({
        "degree_scores": deg_s,
        "lcc_scores":    lcc_s,
        "bc_scores":     bc_s,
        "cc_scores":     cc_s,
    })
    struct_file = os.path.join(out_dir, f"structural_{dataset_name}_{arch}_seed{seed}.csv")
    struct_df.to_csv(struct_file, index=False)

    print(f"  ✓ Saved {metric_file}")
    print(f"  ✓ Saved {struct_file}")

    summary = {
        "fgd":      float(np.mean(fgd_list)),
        "euclid":   float(np.mean(euclid_l)),
        "mmd":      float(np.mean(mmd_l)),
        "kernel":   float(np.mean(kernel_l)),
        "spectral": float(np.mean(spectral_l)),
        "degree":   float(np.mean(deg_s)),
        "lcc":      float(np.mean(lcc_s)),
        "bc":       float(np.mean(bc_s)),
        "cc":       float(np.mean(cc_s)),
    }
    print("\n  Summary metrics (means):")
    for k, v in summary.items():
        print(f"    {k:12s}: {v:.4f}")

    return summary


def aggregate_results(cfg):
    """Aggregate all per-seed CSVs into a single summary text file."""
    out_dir = cfg["output_root"]
    arch    = cfg["arch"]
    files   = glob.glob(os.path.join(out_dir, "*.csv"))
    summary_path = os.path.join(out_dir, f"{arch}_metrics_summary.txt")

    with open(summary_path, "w") as f:
        f.write(f"Metric Summary – arch={arch}\n")
        f.write("=" * 50 + "\n\n")

        for dataset_name in cfg["datasets"]:
            # Metric CSVs
            pattern = os.path.join(out_dir, f"{dataset_name}_{arch}_seed*.csv")
            dfs = [pd.read_csv(fp) for fp in glob.glob(pattern)]
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                s = pd.DataFrame({"Mean": combined.mean(), "Std": combined.std()})
                f.write(f"Dataset: {dataset_name} – Distance metrics\n")
                f.write(s.to_string())
                f.write("\n\n")

            # Structural CSVs
            pattern2 = os.path.join(out_dir, f"structural_{dataset_name}_{arch}_seed*.csv")
            dfs2 = [pd.read_csv(fp) for fp in glob.glob(pattern2)]
            if dfs2:
                combined2 = pd.concat(dfs2, ignore_index=True)
                s2 = pd.DataFrame({"Mean": combined2.mean(), "Std": combined2.std()})
                f.write(f"Dataset: {dataset_name} – Structural metrics\n")
                f.write(s2.to_string())
                f.write("\n\n")

            f.write("-" * 50 + "\n\n")

    print(f"\n✓ Aggregated results saved to {summary_path}")


# ===========================================================================
# 9.  Entry-point
# ===========================================================================

def main():
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = resolve_config(args)

    if args.dry_run:
        import json
        print("\n── Resolved configuration ──")
        print(json.dumps({k: v for k, v in cfg.items() if k != "gin"}, indent=2, default=str))
        print("\n── GIN classifier config ──")
        print(json.dumps(cfg["gin"], indent=2))
        print("\nDry-run: exiting without training.")
        sys.exit(0)

    print(f"\nDevice : {cfg['device']}")
    print(f"Arch   : {cfg['arch']}  ({ARCH_CONFIGS[cfg['arch']]['description']})")
    print(f"Datasets: {cfg['datasets']}")
    print(f"Seeds  : {cfg['seeds']}\n")

    os.makedirs(cfg["model_root"],  exist_ok=True)
    os.makedirs(cfg["output_root"], exist_ok=True)

    all_results = {}
    for dataset_name in cfg["datasets"]:
        for seed in cfg["seeds"]:
            key = (dataset_name, seed)
            all_results[key] = run_experiment(cfg, dataset_name, seed)

    aggregate_results(cfg)
    print("\n✓ All experiments complete.")


if __name__ == "__main__":
    main()