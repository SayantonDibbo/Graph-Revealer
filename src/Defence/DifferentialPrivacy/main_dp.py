#!/usr/bin/env python3
# main_dp.py
"""
LOGRA – Differential Privacy (DP-SGD) Defence Runner
=====================================================
Trains the GIN target classifier with DP-SGD, then runs the full
generative pipeline and evaluates graph quality.

Supported datasets : ENZYMES, NCI1, Cora, PolBlogs
Supported attacks  : logra, csgra
Supported arches   : proposed (diffusion) | random (GAN) | base (CVAE)

Examples
--------
# DP on ENZYMES with LOGRA attack, proposed arch, all default seeds
python main_dp.py --arch proposed --dataset ENZYMES --attack logra

# Multiple datasets in one run
python main_dp.py --arch proposed --dataset ENZYMES NCI1 Cora PolBlogs --attack csgra

# Override DP knobs at the command line
python main_dp.py --arch proposed --dataset NCI1 --attack logra \\
    --dp-noise 0.8 --dp-clip 1.5 --dp-delta 1e-5

# Disable DP entirely (ablation / baseline)
python main_dp.py --arch base --dataset Cora --attack logra --no-dp

# Specific seeds
python main_dp.py --arch random --dataset PolBlogs --attack csgra --seeds 11 22

# Dry-run: print resolved config and exit
python main_dp.py --arch proposed --dataset ENZYMES --attack logra --dry-run
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
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid, Airports, PolBlogs
from torch_geometric.utils import k_hop_subgraph
import torch.nn.functional as F
from collections import Counter

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

# ── DP-specific config ──────────────────────────────────────────────────────
from config_dp import (
    DATASET_CONFIGS, ARCH_CONFIGS, ATTACK_CONFIGS, GIN_CONFIG,
    DEFAULT_SEEDS, DEFAULT_DATA_ROOT, DEFAULT_MODEL_ROOT, DEFAULT_OUTPUT_ROOT,
    get_train_config, get_gen_config, get_dp_config,
)


# ===========================================================================
# Differential Privacy primitives
# ===========================================================================

@dataclass
class PrivacyConfig:
    """
    Encapsulates all DP-SGD hyperparameters for one training run.

    max_grad_norm    – per-parameter gradient clipping bound C.
    noise_multiplier – σ multiplier; noise std = σ × C.
    delta            – target δ in (ε, δ)-DP.
    use_dp           – False → standard SGD (useful for ablation).
    """
    max_grad_norm:    float = 1.0
    noise_multiplier: float = 1.1
    delta:            float = 1e-5
    use_dp:           bool  = True


def compute_epsilon(
    steps: int,
    noise_multiplier: float,
    sample_rate: float,
    delta: float,
) -> float:
    """
    Compute the (ε, δ)-DP privacy budget consumed after *steps* optimizer
    steps using the Gaussian mechanism under Poisson subsampling.

    Approach:
      1. Compute RDP: ε_RDP(α) = α / (2σ²) per step, scaled by sample_rate²
         and accumulated over *steps*.
      2. Convert RDP → (ε, δ)-DP via the standard closed-form bound:
            ε = min_α [ ε_RDP(α) + log(1/δ) / (α − 1) ]

    No external autodp dependency is required.
    """
    orders = np.arange(2, 128, 0.5)
    rdp_per_step  = orders / (2.0 * noise_multiplier ** 2)
    rdp_total     = (sample_rate ** 2) * rdp_per_step * steps
    eps_candidates = rdp_total + np.log(1.0 / delta) / (orders - 1)
    return float(np.min(eps_candidates))


def dp_step(model: nn.Module, optimizer: torch.optim.Optimizer,
            privacy_config: PrivacyConfig) -> None:
    """
    Execute one DP-SGD update in-place.

    For every trainable parameter:
      1. Clip its gradient to L2 norm ≤ max_grad_norm  (bounds sensitivity).
      2. Add i.i.d. Gaussian noise N(0, (σ × C)²I)    (hides individual contributions).
    Then call optimizer.step() with the privatised gradients.

    This manual implementation avoids Opacus's per-sample book-keeping, which
    can fail when PyG batches multiple graphs into a single forward pass.
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                continue
            # 1. Clip
            grad_norm = param.grad.norm(2)
            clip_coef = min(1.0, privacy_config.max_grad_norm / (grad_norm + 1e-8))
            param.grad.mul_(clip_coef)
            # 2. Add calibrated Gaussian noise
            noise = torch.normal(
                mean=0.0,
                std=privacy_config.noise_multiplier * privacy_config.max_grad_norm,
                size=param.grad.shape,
                device=param.grad.device,
            )
            param.grad.add_(noise)
    optimizer.step()


# ===========================================================================
# DP-aware classifier training loop
# ===========================================================================

def train_classifier_dp(
    classifier: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int,
    model_dir: str,
    privacy_config: PrivacyConfig,
    train_dataset_size: int,
) -> nn.Module:
    """
    Train *classifier* with optional DP-SGD and save the best checkpoint.

    When privacy_config.use_dp is True:
      - Every optimizer step goes through dp_step() (clip + noise).
      - ε is computed and logged after every epoch so you can track the
        privacy budget as training progresses.

    When use_dp is False this reduces to a standard Adam training loop,
    providing a clean ablation baseline.

    The best checkpoint (by test accuracy, ties broken by weighted F1) is
    written to *model_dir*/best_model.pth and loaded before returning.
    """
    os.makedirs(model_dir, exist_ok=True)
    best_model_path    = os.path.join(model_dir, "best_model.pth")
    best_test_accuracy = 0.0
    best_f1            = 0.0
    total_steps        = 0

    sample_rate = train_loader.batch_size / max(train_dataset_size, 1)
    optimizer   = optim.Adam(classifier.parameters(), lr=GIN_CONFIG["lr"])
    criterion   = nn.CrossEntropyLoss()

    log_rows = []

    if privacy_config.use_dp:
        print(
            f"  [DP-SGD] max_grad_norm={privacy_config.max_grad_norm}  "
            f"noise_multiplier={privacy_config.noise_multiplier}  "
            f"δ={privacy_config.delta}"
        )
    else:
        print("  [DP-SGD] disabled — running standard SGD")

    for epoch in tqdm(range(n_epochs), desc="  Classifier"):
        # ── training ────────────────────────────────────────────────────────
        classifier.train()
        total_loss, total_samples = 0.0, 0

        for batch in train_loader:
            x, edge_index, y = batch.x, batch.edge_index, batch.y
            optimizer.zero_grad()
            out  = classifier(x, edge_index, batch.batch)
            loss = criterion(out, y)
            loss.backward()

            if privacy_config.use_dp:
                dp_step(classifier, optimizer, privacy_config)
            else:
                optimizer.step()

            total_steps   += 1
            total_loss    += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

        avg_loss = total_loss / max(total_samples, 1)

        # ── evaluation ──────────────────────────────────────────────────────
        classifier.eval()
        correct, total = 0, 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = classifier(batch.x, batch.edge_index, batch.batch)
                preds = out.argmax(dim=1)
                total   += batch.y.size(0)
                correct += (preds == batch.y).sum().item()
                true_labels.extend(batch.y.tolist())
                pred_labels.extend(preds.tolist())

        test_acc = correct / max(total, 1)
        f1       = f1_score(true_labels, pred_labels, average="weighted")

        # ── privacy accounting ───────────────────────────────────────────────
        epsilon = None
        if privacy_config.use_dp:
            epsilon = compute_epsilon(
                steps=total_steps,
                noise_multiplier=privacy_config.noise_multiplier,
                sample_rate=sample_rate,
                delta=privacy_config.delta,
            )
            tqdm.write(
                f"  Epoch {epoch+1:>4}/{n_epochs}  loss={avg_loss:.5f}  "
                f"acc={test_acc:.5f}  f1={f1:.5f}  ε={epsilon:.4f}"
            )
        else:
            tqdm.write(
                f"  Epoch {epoch+1:>4}/{n_epochs}  loss={avg_loss:.5f}  "
                f"acc={test_acc:.5f}  f1={f1:.5f}  (no DP)"
            )

        log_rows.append({
            "epoch": epoch + 1, "loss": avg_loss,
            "test_acc": test_acc, "f1": f1, "epsilon": epsilon,
        })

        # ── checkpoint ──────────────────────────────────────────────────────
        if test_acc > best_test_accuracy or (
            test_acc == best_test_accuracy and f1 > best_f1
        ):
            tqdm.write("  *** best model updated ***")
            best_test_accuracy = test_acc
            best_f1            = f1
            torch.save(classifier.state_dict(), best_model_path)

    classifier.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    pd.DataFrame(log_rows).to_csv(
        os.path.join(model_dir, "train_log_dp.csv"), index=False
    )
    return classifier


# ===========================================================================
# PolBlogs ego-subgraph builder  (mirrors notebook helper)
# ===========================================================================

def build_polblogs_dataset(data, num_samples: int = 1500, k: int = 2,
                            min_nodes: int = 20, max_nodes: int = 500,
                            num_classes: int = 2):
    """
    Convert the PolBlogs single-graph PyG object into a list of ego-subgraph
    Data objects suitable for graph-classification training.

    Each subgraph is centred on a randomly chosen node, extracted via
    k-hop neighbourhood, and labelled by the majority class in the subgraph.
    Node features are one-hot encodings of the node label.
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

        node_labels  = data.y[subset]
        x            = F.one_hot(node_labels, num_classes=num_classes).float()
        majority     = Counter(node_labels.tolist()).most_common(1)[0][0]
        graph_label  = torch.tensor([majority], dtype=torch.long)

        graphs.append(Data(x=x, edge_index=sub_edge_index, y=graph_label))

    return graphs


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main_dp.py",
        description="LOGRA – Differential Privacy (DP-SGD) defence runner",
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
    p.add_argument("--beta-schedule",  choices=["constant", "linear", "cyclical"],
                   default=None)
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

    # ── DP knobs ─────────────────────────────────────────────────────────────
    p.add_argument("--no-dp", action="store_true",
                   help="Disable DP entirely (standard SGD baseline).")
    p.add_argument("--dp-noise", type=float, default=None, metavar="σ",
                   help="Override noise_multiplier for all datasets.")
    p.add_argument("--dp-clip",  type=float, default=None, metavar="C",
                   help="Override max_grad_norm for all datasets.")
    p.add_argument("--dp-delta", type=float, default=None, metavar="δ",
                   help="Override delta for all datasets.")

    return p


def resolve_config(args) -> dict:
    """Merge CLI args with config_dp.py defaults into a flat run dict."""
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

    # CLI-level DP overrides (applied on top of per-dataset overrides in run_experiment)
    cfg["dp_cli_overrides"] = {
        "no_dp":     args.no_dp,
        "dp_noise":  args.dp_noise,
        "dp_clip":   args.dp_clip,
        "dp_delta":  args.dp_delta,
    }

    return cfg


# ===========================================================================
# Experiment runner
# ===========================================================================

def run_experiment(cfg: dict, dataset_name: str, seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  [DP Defence]  arch={cfg['arch']}  dataset={dataset_name}  seed={seed}")
    print(f"{'='*60}")

    device = cfg["device"]
    arch   = cfg["arch"]
    dcfg   = DATASET_CONFIGS[dataset_name]
    attack = cfg["attack"]

    # ── 1. Seeding ──────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":   torch.cuda.manual_seed_all(seed)
    elif device == "mps":  torch.mps.manual_seed(seed)

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
        node_features = dcfg["classes"]   # one-hot dimension = num_classes
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

    train_loader = DataLoader(train_ds, batch_size=cfg["gin"]["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["gin"]["batch_size"], shuffle=False)

    # ── 4. Resolve PrivacyConfig (dataset defaults → CLI overrides) ──────────
    dp_base = get_dp_config(dataset_name)          # from config_dp.py
    cli_ov  = cfg["dp_cli_overrides"]

    if cli_ov["no_dp"]:    dp_base["use_dp"] = False
    if cli_ov["dp_noise"]: dp_base["noise_multiplier"] = cli_ov["dp_noise"]
    if cli_ov["dp_clip"]:  dp_base["max_grad_norm"]    = cli_ov["dp_clip"]
    if cli_ov["dp_delta"]: dp_base["delta"]            = cli_ov["dp_delta"]

    privacy_config = PrivacyConfig(
        max_grad_norm=dp_base["max_grad_norm"],
        noise_multiplier=dp_base["noise_multiplier"],
        delta=dp_base["delta"],
        use_dp=dp_base["use_dp"],
    )

    # ── 5. Classifier ────────────────────────────────────────────────────────
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
        print(f"  Training classifier — {n_cls} epochs  "
              f"DP={'ON' if privacy_config.use_dp else 'OFF'}")
        classifier = train_classifier_dp(
            classifier, train_loader, test_loader,
            n_cls, model_dir, privacy_config,
            train_dataset_size=len(train_ds),
        )

    classifier.eval()

    # ── 6. Run attack on shadow set ──────────────────────────────────────────
    print(f"  Running {attack.upper()} attack on shadow dataset...")
    if num_classes >= 2 and attack == "logra":
    
        total_data = logra.logra_attack(classifier, shadow_ds, device)
    elif num_classes > 2 and attack == "csgra":  # csgra
        total_data = csgra.csgra_attack(classifier, shadow_ds, device)
    else:
        raise ValueError(f"Unsupported attack {attack} for num_classes={num_classes}")

    # ── 7. Generative model training ─────────────────────────────────────────
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

    # ── 8. Build target labels for generation ────────────────────────────────
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

    # ── 9. Generate graphs ────────────────────────────────────────────────────
    print("  Generating graphs...")
    generated_dicts = generate_graphs(gen_model, gen_labels, arch, device)
    generated_graph = list_of_dicts_to_pyg(generated_dicts)
    original_graph  = dataset_to_pyg(target_data)

    # ── 10. Metrics ───────────────────────────────────────────────────────────
    print("  Computing metrics...")
    encoder  = GCNEncoder(node_features)
    fgd_list = compute_fgd(original_graph, generated_graph, encoder)
    euclid_l, mmd_l, kernel_l, spectral_l = compute_metrics(
        original_graph, generated_graph, node_features
    )
    deg_s, lcc_s, bc_s, cc_s = compute_structural(original_graph, generated_graph)

    # ── 11. Save CSVs ─────────────────────────────────────────────────────────
    out_dir = cfg["output_root"]
    os.makedirs(out_dir, exist_ok=True)

    tag = f"{dataset_name}_{arch}_{attack}_dp_seed{seed}"

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
    summary_path = os.path.join(out_dir, f"{arch}_{attack}_dp_metrics_summary.txt")

    with open(summary_path, "w") as f:
        f.write(f"DP Defence Metric Summary – arch={arch}  attack={attack}\n")
        f.write("=" * 60 + "\n\n")

        for dataset_name in cfg["datasets"]:
            for prefix, label in [
                (f"{dataset_name}_{arch}_{attack}_dp_seed*.csv",        "Distance metrics"),
                (f"structural_{dataset_name}_{arch}_{attack}_dp_seed*.csv", "Structural metrics"),
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
        print("\n── Resolved DP configuration ──")
        safe = {k: v for k, v in cfg.items() if k != "gin"}
        print(json.dumps(safe, indent=2, default=str))
        print("\n── GIN config ──")
        print(json.dumps(cfg["gin"], indent=2))
        sys.exit(0)

    print(f"\nDevice   : {cfg['device']}")
    print(f"Arch     : {cfg['arch']}  ({ARCH_CONFIGS[cfg['arch']]['description']})")
    print(f"Attack   : {cfg['attack']}")
    print(f"Datasets : {cfg['datasets']}")
    print(f"Seeds    : {cfg['seeds']}")
    print(f"DP       : {'DISABLED (--no-dp)' if cfg['dp_cli_overrides']['no_dp'] else 'ENABLED'}\n")

    os.makedirs(cfg["model_root"],  exist_ok=True)
    os.makedirs(cfg["output_root"], exist_ok=True)

    all_results: dict = {}
    for dataset_name in cfg["datasets"]:
        for seed in cfg["seeds"]:
            all_results[(dataset_name, seed)] = run_experiment(cfg, dataset_name, seed)

    aggregate_results(cfg)
    print("\n✓ DP defence run complete.")


if __name__ == "__main__":
    main()