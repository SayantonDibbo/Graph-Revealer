#!/usr/bin/env python3
# main_pe.py
"""
LOGRA – Pseudo-Edge / Edge-Perturbation Defence Runner
======================================================
Trains the GIN target classifier with pseudo-edge augmentation applied to
the training graphs, then runs the full generative pipeline and evaluates
graph quality.

The defence: before training, synthetic edges are inserted between nodes
whose feature vectors are most similar (cosine or euclidean), making the
classifier more robust to structural membership-inference attacks.  Test
graphs are NEVER modified so evaluation always reflects real structure.

Supported datasets : ENZYMES, NCI1, Cora, PolBlogs
Supported attacks  : logra, csgra
Supported arches   : proposed (diffusion) | random (GAN) | base (CVAE)

Examples
--------
# Pseudo-edge defence on ENZYMES, LOGRA attack, proposed arch
python main_ep.py --arch proposed --dataset ENZYMES --attack logra

# Multiple datasets in one run
python main_ep.py --arch proposed --dataset ENZYMES NCI1 Cora PolBlogs --attack csgra

# Override augmentation knobs at the command line
python main_ep.py --arch proposed --dataset Cora --attack logra \\
    --pe-top-k 10 --pe-similarity euclidean

# Disable augmentation entirely (standard training baseline / ablation)
python main_ep.py --arch base --dataset NCI1 --attack csgra --no-pseudo-edges

# Specific seeds
python main_ep.py --arch random --dataset PolBlogs --attack logra --seeds 11 22

# Dry-run: print resolved config and exit
python main_ep.py --arch proposed --dataset ENZYMES --attack logra --dry-run
"""

import argparse
import copy
import glob
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")

from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid, Airports, PolBlogs
from torch_geometric.utils import k_hop_subgraph, to_dense_adj

# ── shared project imports ──────────────────────────────────────────────────
from src.attack.LOGRA import logra
from src.attack.CSGRA import csgra
from src.utils.utils import (
    load_dataset, infer_node_features, dataset_to_pyg, list_of_dicts_to_pyg,
    GraphDatasetWrapper,
)
from src.models.gnn import GINClassification
from src.models.gan import GraphGAN
from src.models.autoencoder import GraphCVAE
from src.models.diffusion import GraphDiffusionModel
from src.metrics.metrics import compute_fgd, compute_metrics, compute_structural, GCNEncoder
from inference.inference import generate_graphs
from train.train_gan import train_gan
from train.train_autoencoder import train_cvae
from train.train_diffusion import train_diffusion
from train.train_target import train_classifier

# ── PE-specific config ───────────────────────────────────────────────────────
from config_ep import (
    DATASET_CONFIGS, ARCH_CONFIGS, ATTACK_CONFIGS, GIN_CONFIG,
    DEFAULT_SEEDS, DEFAULT_DATA_ROOT, DEFAULT_MODEL_ROOT, DEFAULT_OUTPUT_ROOT,
    get_train_config, get_gen_config, get_pseudo_edge_config,
)


# ===========================================================================
# Pseudo-edge augmentation primitives
# ===========================================================================

@dataclass
class PseudoEdgeConfig:
    """
    Encapsulates all pseudo-edge augmentation hyperparameters.

    top_k            – edges added per node (per_node_topk=True) or
                       globally (per_node_topk=False).
    similarity       – 'cosine' or 'euclidean'.
    per_node_topk    – balanced (True) vs. global budget (False).
    min_nodes        – skip graphs smaller than this.
    use_pseudo_edges – master on/off switch.
    """
    top_k:            int  = 5
    similarity:       str  = "cosine"
    per_node_topk:    bool = True
    min_nodes:        int  = 4
    use_pseudo_edges: bool = True


def _compute_similarity_matrix(X: torch.Tensor, method: str) -> torch.Tensor:
    """
    Compute an N×N pairwise similarity matrix for node feature matrix X.

    cosine    – normalise rows, then dot-product; ∈ [-1, 1].
                Higher = more similar direction in feature space.
    euclidean – 1 / (1 + ||x−y||²); ∈ (0, 1].
                Higher = closer in feature space.
    """
    if method == "cosine":
        norms  = X.norm(dim=1, keepdim=True).clamp(min=1e-8)
        X_norm = X / norms
        return torch.mm(X_norm, X_norm.t())
    elif method == "euclidean":
        sq    = (X ** 2).sum(dim=1, keepdim=True)
        dist2 = (sq + sq.t() - 2 * torch.mm(X, X.t())).clamp(min=0)
        return 1.0 / (1.0 + dist2)
    else:
        raise ValueError(
            f"Unknown similarity: {method!r}. Choose 'cosine' or 'euclidean'."
        )


def _add_pseudo_edges(data: Data, config: PseudoEdgeConfig) -> Data:
    """
    Add synthetic edges to a single graph based on node-feature similarity.

    Algorithm:
      1. Compute pairwise similarity between all node feature vectors.
      2. Zero out the diagonal and existing edges (only propose NEW connections).
      3. For each node (per_node_topk) or globally, select the top-k most
         similar unconnected pairs.
      4. Return a new Data object with both the original and new edges.

    The original Data object is not mutated.
    Only called on training graphs — test graphs are passed through unchanged.
    """
    N = data.num_nodes
    if N < config.min_nodes:
        return data

    X   = data.x.float()
    sim = _compute_similarity_matrix(X, method=config.similarity)
    sim.fill_diagonal_(0)

    # Zero out already-existing edges so we never duplicate them
    adj = to_dense_adj(data.edge_index, max_num_nodes=N)[0]
    sim = sim * (1.0 - adj)

    new_src, new_dst = [], []

    if config.per_node_topk:
        # Balanced: each node independently nominates its top-k candidates.
        # This distributes new edges evenly, avoiding high-degree hubs.
        k = min(config.top_k, N - 1)
        _, topk_idxs = torch.topk(sim, k=k, dim=1)
        for src in range(N):
            for dst in topk_idxs[src].tolist():
                if sim[src, dst] > 0:
                    new_src.append(src)
                    new_dst.append(dst)
    else:
        # Global: find the top-k highest similarity scores across the whole
        # N×N matrix. Simpler but may concentrate edges around popular nodes.
        k = min(config.top_k, N * (N - 1))
        flat_topk = torch.topk(sim.view(-1), k=k)
        for flat_idx in flat_topk.indices.tolist():
            src = flat_idx // N
            dst = flat_idx  % N
            if sim[src, dst] > 0:
                new_src.append(src)
                new_dst.append(dst)

    if len(new_src) == 0:
        return data  # nothing to add (fully connected or all sims were 0)

    # Build undirected new edges (both directions)
    new_src_t = torch.tensor(new_src + new_dst, dtype=torch.long)
    new_dst_t = torch.tensor(new_dst + new_src, dtype=torch.long)
    new_edge  = torch.stack([new_src_t, new_dst_t], dim=0)

    aug_edge_index = torch.cat([data.edge_index, new_edge], dim=1)
    return Data(x=data.x, edge_index=aug_edge_index, y=data.y)


def apply_pseudo_edges(dataset, config: PseudoEdgeConfig,
                       split_name: str = "train"):
    """
    Apply pseudo-edge augmentation to every graph in a dataset split.

    Key invariant: this function ONLY modifies training graphs.
    Passing split_name='test' (or any value other than 'train') is a no-op,
    guaranteeing that evaluation always uses the real, unaugmented topology.

    Prints a summary of how many edges were added so you can verify the
    augmentation worked as intended.
    """
    if not config.use_pseudo_edges or split_name != "train":
        return dataset

    before    = sum(g.edge_index.size(1) for g in dataset)
    augmented = [_add_pseudo_edges(g, config) for g in dataset]
    after     = sum(g.edge_index.size(1) for g in augmented)
    added     = after - before

    print(
        f"\n  [Pseudo Edges] Applied to {len(dataset)} training graphs.\n"
        f"    similarity  : {config.similarity}\n"
        f"    mode        : {'per-node top-' if config.per_node_topk else 'global top-'}"
        f"{config.top_k}\n"
        f"    min_nodes   : {config.min_nodes}\n"
        f"    edges before: {before:,}\n"
        f"    edges after : {after:,}\n"
        f"    edges added : {added:,}  (+{100*added/max(before,1):.1f}%)\n"
        f"    test set    : UNTOUCHED\n"
    )
    return augmented


# ===========================================================================
# Standard classifier training (no DP)
# ===========================================================================

def train_classifier_standard(
    classifier: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int,
    model_dir: str,
) -> nn.Module:
    """
    Train *classifier* with standard Adam, save the best checkpoint.

    Best checkpoint is selected by highest test accuracy, ties broken by
    weighted F1.  Training log is written to *model_dir*/train_log_pe.csv.
    """
    os.makedirs(model_dir, exist_ok=True)
    best_model_path    = os.path.join(model_dir, "best_model.pth")
    best_test_accuracy = 0.0
    best_f1            = 0.0

    optimizer = optim.Adam(classifier.parameters(), lr=GIN_CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    log_rows  = []

    for epoch in tqdm(range(n_epochs), desc="  Classifier"):
        # ── train ────────────────────────────────────────────────────────────
        classifier.train()
        total_loss, total_samples = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out  = classifier(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss    += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

        avg_loss = total_loss / max(total_samples, 1)

        # ── evaluate ─────────────────────────────────────────────────────────
        classifier.eval()
        correct, total = 0, 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                preds  = classifier(batch.x, batch.edge_index, batch.batch).argmax(1)
                total  += batch.y.size(0)
                correct += (preds == batch.y).sum().item()
                true_labels.extend(batch.y.tolist())
                pred_labels.extend(preds.tolist())

        test_acc = correct / max(total, 1)
        f1       = f1_score(true_labels, pred_labels, average="weighted")
        tqdm.write(
            f"  Epoch {epoch+1:>4}/{n_epochs}  loss={avg_loss:.5f}  "
            f"acc={test_acc:.5f}  f1={f1:.5f}"
        )

        log_rows.append({"epoch": epoch+1, "loss": avg_loss,
                         "test_acc": test_acc, "f1": f1})

        if test_acc > best_test_accuracy or (
            test_acc == best_test_accuracy and f1 > best_f1
        ):
            tqdm.write("  *** best model updated ***")
            best_test_accuracy = test_acc
            best_f1            = f1
            torch.save(classifier.state_dict(), best_model_path)

    classifier.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    pd.DataFrame(log_rows).to_csv(
        os.path.join(model_dir, "train_log_pe.csv"), index=False
    )
    return classifier


# ===========================================================================
# PolBlogs ego-subgraph builder
# ===========================================================================

def build_polblogs_dataset(data, num_samples: int = 1500, k: int = 2,
                            min_nodes: int = 20, max_nodes: int = 500,
                            num_classes: int = 2):
    """
    Convert the PolBlogs single-graph PyG object into a list of ego-subgraph
    Data objects.  Node features are one-hot encodings of the node label;
    graph label is the majority class within the subgraph.
    """
    graphs  = []
    n_nodes = data.num_nodes

    while len(graphs) < num_samples:
        center = random.randint(0, n_nodes - 1)
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            center, k, data.edge_index, relabel_nodes=True
        )
        n = subset.size(0)
        if n < min_nodes or n > max_nodes:
            continue

        node_labels = data.y[subset]
        x           = F.one_hot(node_labels, num_classes=num_classes).float()
        majority    = Counter(node_labels.tolist()).most_common(1)[0][0]
        graphs.append(Data(
            x=x,
            edge_index=sub_edge_index,
            y=torch.tensor([majority], dtype=torch.long),
        ))

    return graphs


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main_pe.py",
        description="LOGRA – Pseudo-Edge (edge perturbation) defence runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── what to run ──────────────────────────────────────────────────────────
    p.add_argument("--arch", required=True, choices=list(ARCH_CONFIGS.keys()),
                   help="Generative architecture.")
    p.add_argument("--dataset", required=True, nargs="+",
                   choices=list(DATASET_CONFIGS.keys()),
                   help="One or more datasets.")
    p.add_argument("--attack", required=True, choices=list(ATTACK_CONFIGS.keys()),
                   help="Attack type: logra | csgra")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   metavar="SEED", help="Random seeds.")

    # ── epoch overrides ──────────────────────────────────────────────────────
    p.add_argument("--epochs-classifier", type=int, default=None, metavar="N")
    p.add_argument("--epochs-gen",        type=int, default=None, metavar="N")

    # ── generative model ─────────────────────────────────────────────────────
    p.add_argument("--gen-lr",         type=float, default=None)
    p.add_argument("--gen-batch-size", type=int,   default=None)
    p.add_argument("--latent-dim",     type=int,   default=None)
    p.add_argument("--hidden-dim",     type=int,   default=None)
    p.add_argument("--beta-schedule",
                   choices=["constant", "linear", "cyclical"], default=None)
    p.add_argument("--lr-g", type=float, default=None)
    p.add_argument("--lr-d", type=float, default=None)

    # ── paths ────────────────────────────────────────────────────────────────
    p.add_argument("--data-root",   default=DEFAULT_DATA_ROOT)
    p.add_argument("--model-root",  default=DEFAULT_MODEL_ROOT)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    # ── flags ────────────────────────────────────────────────────────────────
    p.add_argument("--skip-classifier", action="store_true")
    p.add_argument("--skip-gen",        action="store_true")
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("--device",          default=None)

    # ── pseudo-edge knobs ────────────────────────────────────────────────────
    p.add_argument("--no-pseudo-edges", action="store_true",
                   help="Disable augmentation entirely (standard training baseline).")
    p.add_argument("--pe-top-k", type=int, default=None, metavar="K",
                   help="Override top-k for all datasets (default per dataset in config).")
    p.add_argument("--pe-similarity", choices=["cosine", "euclidean"], default=None,
                   help="Override similarity metric for all datasets.")
    p.add_argument("--pe-global", action="store_true",
                   help="Use global top-k budget instead of per-node top-k.")

    return p


def resolve_config(args) -> dict:
    """Merge CLI args with config_pe.py defaults into a flat run dict."""
    cfg = {
        "arch":    args.arch,
        "datasets": args.dataset,
        "seeds":   args.seeds,
        "attack":  args.attack,
    }

    # Device
    if args.device:
        cfg["device"] = args.device
    elif torch.cuda.is_available():
        cfg["device"] = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cfg["device"] = "mps"
    else:
        cfg["device"] = "cpu"

    cfg.update({
        "data_root":   args.data_root,
        "model_root":  args.model_root,
        "output_root": args.output_root,
        "skip_classifier": args.skip_classifier,
        "skip_gen":        args.skip_gen,
        "gin":             dict(GIN_CONFIG),
        "epochs_classifier_override": args.epochs_classifier,
        "epochs_gen_override":        args.epochs_gen,
    })

    # Generative model config
    gen_cfg = get_gen_config(args.arch)
    if args.gen_lr         is not None: gen_cfg["lr"] = gen_cfg["lr_g"] = gen_cfg["lr_d"] = args.gen_lr
    if args.gen_batch_size is not None: gen_cfg["batch_size"] = args.gen_batch_size
    if args.latent_dim     is not None: gen_cfg["latent_dim"] = args.latent_dim
    if args.hidden_dim     is not None: gen_cfg["hidden_dim"] = args.hidden_dim
    if args.beta_schedule  is not None and "beta_schedule" in gen_cfg:
        gen_cfg["beta_schedule"] = args.beta_schedule
    if args.lr_g is not None and "lr_g" in gen_cfg: gen_cfg["lr_g"] = args.lr_g
    if args.lr_d is not None and "lr_d" in gen_cfg: gen_cfg["lr_d"] = args.lr_d
    cfg["gen"] = gen_cfg

    # CLI-level PE overrides (applied on top of per-dataset overrides in run_experiment)
    cfg["pe_cli_overrides"] = {
        "no_pseudo_edges": args.no_pseudo_edges,
        "pe_top_k":        args.pe_top_k,
        "pe_similarity":   args.pe_similarity,
        "pe_global":       args.pe_global,
    }

    return cfg


# ===========================================================================
# Experiment runner
# ===========================================================================

def run_experiment(cfg: dict, dataset_name: str, seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  [PE Defence]  arch={cfg['arch']}  dataset={dataset_name}  seed={seed}")
    print(f"{'='*60}")

    device = cfg["device"]
    arch   = cfg["arch"]
    dcfg   = DATASET_CONFIGS[dataset_name]
    attack = cfg["attack"]

    # ── 1. Seeding ──────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":  torch.cuda.manual_seed_all(seed)
    elif device == "mps": torch.mps.manual_seed(seed)

    # ── 2. Load / build dataset ──────────────────────────────────────────────
    print("  Loading dataset...")
    if dataset_name == "PolBlogs":
        raw      = PolBlogs(root=os.path.join(cfg["data_root"], "PolBlogs"))[0]
        dataset  = build_polblogs_dataset(
            raw,
            num_samples=1500,
            k=dcfg["k_hop"],
            max_nodes=dcfg["max_nodes"],
            num_classes=dcfg["classes"],
        )
        node_features = dcfg["classes"]   # one-hot width = num_classes
    else:
        dataset       = load_dataset(dataset_name, cfg["data_root"])
        node_features = (
            dcfg["node_features"] if dcfg["node_features"] != -1
            else infer_node_features(dataset)
        )

    num_classes = dcfg["classes"]
    max_nodes   = dcfg["max_nodes"]
    print(f"  node_features={node_features}  num_classes={num_classes}  max_nodes={max_nodes}")

    # ── 3. Split ─────────────────────────────────────────────────────────────
    target_ds, shadow_ds = train_test_split(dataset, test_size=0.5, random_state=42)
    train_ds,  test_ds   = train_test_split(target_ds, test_size=0.2, random_state=42)

    # ── 4. Resolve PseudoEdgeConfig (dataset defaults → CLI overrides) ────────
    pe_base = get_pseudo_edge_config(dataset_name)   # from config_pe.py
    cli_ov  = cfg["pe_cli_overrides"]

    if cli_ov["no_pseudo_edges"]: pe_base["use_pseudo_edges"] = False
    if cli_ov["pe_top_k"]:        pe_base["top_k"]            = cli_ov["pe_top_k"]
    if cli_ov["pe_similarity"]:   pe_base["similarity"]       = cli_ov["pe_similarity"]
    if cli_ov["pe_global"]:       pe_base["per_node_topk"]    = False

    pe_config = PseudoEdgeConfig(
        top_k=pe_base["top_k"],
        similarity=pe_base["similarity"],
        per_node_topk=pe_base["per_node_topk"],
        min_nodes=pe_base["min_nodes"],
        use_pseudo_edges=pe_base["use_pseudo_edges"],
    )

    # ── 5. Apply pseudo-edge augmentation to training split ONLY ─────────────
    # test_ds is intentionally passed directly to the loader without augmentation.
    train_ds = apply_pseudo_edges(train_ds, pe_config, split_name="train")

    train_loader = DataLoader(train_ds, batch_size=cfg["gin"]["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["gin"]["batch_size"], shuffle=False)

    # ── 6. Classifier ────────────────────────────────────────────────────────
    model_dir = os.path.join(cfg["model_root"], dataset_name, arch, str(seed))
    gin_path  = os.path.join(model_dir, "best_model.pth")

    classifier = GINClassification(
        in_channels=node_features,
        hidden_channels=cfg["gin"]["hidden_channels"],
        out_channels=cfg["gin"]["out_channels"],
        num_classes=num_classes,
    )

    tc    = get_train_config(arch, dataset_name)
    n_cls = cfg["epochs_classifier_override"] or tc["num_epoch_classifier"]
    n_gen = cfg["epochs_gen_override"]        or tc["num_epoch_gen"]

    if cfg["skip_classifier"] and os.path.exists(gin_path):
        print(f"  Loading classifier from {gin_path}")
        classifier.load_state_dict(torch.load(gin_path, map_location="cpu"))
    else:
        aug_status = (
            f"PE ON  (top_k={pe_config.top_k}, sim={pe_config.similarity})"
            if pe_config.use_pseudo_edges else "PE OFF (standard training)"
        )
        print(f"  Training classifier — {n_cls} epochs  {aug_status}")
        classifier = train_classifier_standard(
            classifier, train_loader, test_loader, n_cls, model_dir
        )

    classifier.eval()

    # ── 7. Run attack on shadow set ──────────────────────────────────────────
    print(f"  Running {attack.upper()} attack on shadow dataset...")
    if attack == "logra":
        total_data = logra.logra_attack(classifier, shadow_ds, device)
    else:  # csgra
        total_data = csgra.csgra_attack(classifier, shadow_ds, device)

    # ── 8. Generative model training ─────────────────────────────────────────
    gen_cfg        = cfg["gen"]
    batch_size_gen = gen_cfg.get("batch_size", 32)
    gen_dataset    = GraphDatasetWrapper(total_data, max_nodes, num_classes)
    gen_loader     = DataLoader(gen_dataset, batch_size=batch_size_gen,
                                shuffle=True, num_workers=0)
    gen_model_path = os.path.join(model_dir, "gen_model.pth")

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
            torch.save({"generator":     gen_model.generator.state_dict(),
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

    # ── 9. Build target labels for generation ────────────────────────────────
    print("  Building target labels...")
    gen_labels, target_data = [], []
    for data in target_ds:
        data = data.to("cpu")
        with torch.no_grad():
            pred = classifier(data.x, data.edge_index, None).argmax(dim=1).item()
        cloned = copy.deepcopy(data)
        if pred != data.y.item():
            cloned.y = torch.tensor([pred], dtype=torch.long)
        target_data.append(cloned)
        gen_labels.append(pred)

    # ── 10. Generate graphs ───────────────────────────────────────────────────
    print("  Generating graphs...")
    generated_dicts = generate_graphs(gen_model, gen_labels, arch, device)
    generated_graph = list_of_dicts_to_pyg(generated_dicts)
    original_graph  = dataset_to_pyg(target_data)

    # ── 11. Metrics ───────────────────────────────────────────────────────────
    print("  Computing metrics...")
    encoder  = GCNEncoder(node_features)
    fgd_list = compute_fgd(original_graph, generated_graph, encoder)
    euclid_l, mmd_l, kernel_l, spectral_l = compute_metrics(
        original_graph, generated_graph, node_features
    )
    deg_s, lcc_s, bc_s, cc_s = compute_structural(original_graph, generated_graph)

    # ── 12. Save CSVs ─────────────────────────────────────────────────────────
    out_dir = cfg["output_root"]
    os.makedirs(out_dir, exist_ok=True)

    tag = f"{dataset_name}_{arch}_{attack}_pe_seed{seed}"

    pd.DataFrame({
        "fgd_mean": fgd_list, "euclidean_mean": euclid_l,
        "mmd_mean": mmd_l,    "kernel_mean": kernel_l,
        "spectral_mean": spectral_l,
    }).to_csv(os.path.join(out_dir, f"{tag}.csv"), index=False)

    pd.DataFrame({
        "degree_scores": deg_s, "lcc_scores": lcc_s,
        "bc_scores": bc_s,      "cc_scores": cc_s,
    }).to_csv(os.path.join(out_dir, f"structural_{tag}.csv"), index=False)

    print(f"  ✓ Saved {tag}.csv")
    print(f"  ✓ Saved structural_{tag}.csv")

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
    print("\n  Summary means:")
    for k, v in summary.items():
        print(f"    {k:12s}: {v:.4f}")
    return summary


# ===========================================================================
# Aggregation
# ===========================================================================

def aggregate_results(cfg: dict) -> None:
    """Aggregate per-seed CSVs into a single summary text file."""
    out_dir = cfg["output_root"]
    arch    = cfg["arch"]
    attack  = cfg["attack"]
    summary_path = os.path.join(out_dir, f"{arch}_{attack}_pe_metrics_summary.txt")

    with open(summary_path, "w") as f:
        f.write(f"PE Defence Metric Summary – arch={arch}  attack={attack}\n")
        f.write("=" * 60 + "\n\n")

        for dataset_name in cfg["datasets"]:
            for prefix, label in [
                (f"{dataset_name}_{arch}_{attack}_pe_seed*.csv",            "Distance metrics"),
                (f"structural_{dataset_name}_{arch}_{attack}_pe_seed*.csv", "Structural metrics"),
            ]:
                dfs = [pd.read_csv(fp)
                       for fp in glob.glob(os.path.join(out_dir, prefix))]
                if dfs:
                    combined = pd.concat(dfs, ignore_index=True)
                    s = pd.DataFrame({"Mean": combined.mean(), "Std": combined.std()})
                    f.write(f"Dataset: {dataset_name} – {label}\n")
                    f.write(s.to_string())
                    f.write("\n\n")

            f.write("-" * 60 + "\n\n")

    print(f"\n✓ Aggregated results → {summary_path}")


# ===========================================================================
# Entry-point
# ===========================================================================

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = resolve_config(args)

    if args.dry_run:
        import json
        print("\n── Resolved PE configuration ──")
        safe = {k: v for k, v in cfg.items() if k != "gin"}
        print(json.dumps(safe, indent=2, default=str))
        print("\n── GIN config ──")
        print(json.dumps(cfg["gin"], indent=2))
        sys.exit(0)

    ov = cfg["pe_cli_overrides"]
    print(f"\nDevice   : {cfg['device']}")
    print(f"Arch     : {cfg['arch']}  ({ARCH_CONFIGS[cfg['arch']]['description']})")
    print(f"Attack   : {cfg['attack']}")
    print(f"Datasets : {cfg['datasets']}")
    print(f"Seeds    : {cfg['seeds']}")
    print(f"PseudoEdges: {'DISABLED (--no-pseudo-edges)' if ov['no_pseudo_edges'] else 'ENABLED'}")
    if not ov["no_pseudo_edges"]:
        print(f"  top_k     : {ov['pe_top_k'] or 'from config per dataset'}")
        print(f"  similarity: {ov['pe_similarity'] or 'from config per dataset'}\n")

    os.makedirs(cfg["model_root"],  exist_ok=True)
    os.makedirs(cfg["output_root"], exist_ok=True)

    all_results: dict = {}
    for dataset_name in cfg["datasets"]:
        for seed in cfg["seeds"]:
            all_results[(dataset_name, seed)] = run_experiment(cfg, dataset_name, seed)

    aggregate_results(cfg)
    print("\n✓ PE defence run complete.")


if __name__ == "__main__":
    main()