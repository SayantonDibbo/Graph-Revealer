# The Graph Revealer: Generative Diffusion Model Inversion Attacks against Graph Neural Networks

The Figure below shows the framework of The Graph Revealer:
<h1 align="center">
  <img src="Framework/Framework.png" alt="framework" />
</h1>


#### Keywords: Graph Reconstruction Attacks, Generative Diffusion Models, Model Inversion Threats, Privacy Vulnerabilities, Label-only Attacks, Differential Privacy, Edge Perturbation

# LOGRA — Graph Generation Benchmark

A benchmarking framework for revealing target model training data demonstrated using three architectures:
**Laplace Diffusion** (proposed), **GAN**, and **AE**.
Each architecture is evaluated against real graph data using a GIN-based shadow model attack pipeline.
Two defence mechanisms are included: **Differential Privacy (DP-SGD)** and **Edge Perturbation (Pseudo-Edge Augmentation)**.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Attack](#attack-runner-mainpy)
  - [Defence: Differential Privacy](#defence-differential-privacy-main_dppy)
  - [Defence: Edge Perturbation](#defence-edge-perturbation-main_eppy)
- [Output](#output)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)

---

## Overview

Graph data, which captures relational patterns and makes many data mining activities easier, has not received as much attention in the literature as privacy problems in image and tabular data. Relational graph data that contains nodes and edges may be extremely sensitive, creating avenues for attackers to obtain or disclose private graph information. In order to leak or reveal target model training data (graphs), we first propose two novel black-box model inversion (MI) privacy attacks against graph neural networks (GNNs) with limited adversarial capabilities. These attacks are called label-only graph reconstruction attacks (LO-GRA) and confidence score-based graph reconstruction attacks (CSGRA).

We also provide two complementary defence mechanisms that can be used independently or compared against each other:

- **Differential Privacy (DP-SGD):** Trains the GIN classifier with gradient clipping and calibrated Gaussian noise, providing formal (ε, δ)-DP guarantees.
- **Edge Perturbation:** Augments training graphs with synthetic pseudo-edges between feature-similar nodes, making the classifier more robust to structural membership-inference attacks.

---

## Pipeline

```
Full Dataset
    ├── 50% → Target split
    │       ├── 80% → Train set   ──► trains GIN classifier  [+ defence applied here]
    │       └── 20% → Test set    ──► evaluates GIN classifier
    │
    └── 50% → Shadow split
              └── queried through trained GIN classifier
                      ├── Correct predictions  → training data for generative model
                      └── Incorrect predictions → discarded
                                  ↓
                      GAN / Diffusion / CVAE trained on correct shadow graphs
                                  ↓
              Target split queried for predicted labels
                                  ↓
                      Generated graphs evaluated against target graphs
```

### Defence integration points

| Defence | Where it acts | What it changes |
|---|---|---|
| **DP-SGD** | Classifier training loop | Gradient clipping + Gaussian noise per step |
| **Edge Perturbation** | Training graphs only | Adds pseudo-edges between similar nodes before training |


---

## Requirements

- Python >= 3.8
- PyTorch >= 1.13
- PyTorch Geometric >= 2.0
- GraKeL
- NetworkX
- scikit-learn
- scipy
- tqdm
- pandas
- numpy
- matplotlib

All required modules are listed in `requirements.txt`.

### Datasets

| Dataset  | Source    | Type                 |
|----------|-----------|----------------------|
| ENZYMES  | TUDataset | Graph classification |
| NCI1     | TUDataset | Graph classification |
| Cora     | Planetoid | Node classification  |
| PolBlogs | PolBlogs  | Node classification  |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SayantonDibbo/Graph-Revealer.git
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install PyTorch

Visit [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) and select the appropriate command for your system and CUDA version. For example:

```bash
# CPU only
pip install torch torchvision torchaudio

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install PyTorch Geometric

```bash
pip install torch-geometric

# Optional but recommended for full PyG support:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-<YOUR_TORCH_VERSION>+<CUDA>.html
```

Replace `<YOUR_TORCH_VERSION>` (e.g. `2.0.0`) and `<CUDA>` (e.g. `cu118` or `cpu`) accordingly.

### 5. Install remaining dependencies

```bash
pip install grakel networkx scikit-learn scipy tqdm pandas numpy matplotlib
```

### 6. Verify installation

```bash
python -c "import torch; import torch_geometric; import grakel; print('All dependencies OK')"
```

---

## Configuration

Each runner has its own dedicated config file. All dataset and architecture settings are defined there — no need to edit the runner scripts directly.

| Config file | Used by | Purpose |
|---|---|---|
| `config.py` | `main.py` | Attack baseline settings |
| `src/Defence/DifferentialPrivacy/config_dp.py` | `main_dp.py` | DP-SGD hyperparameters |
| `src/Defence/EdgePerturbation/config_ep.py` | `main_ep.py` | Pseudo-edge hyperparameters |

All three config files share the same structure for common sections:

**`DATASET_CONFIGS`** — per-dataset properties: number of classes, max nodes, node feature dimension, k-hop radius (PolBlogs).

**`ARCH_CONFIGS`** — generative architectures (`proposed`, `random`, `base`) with descriptions.

**`GIN_CONFIG`** — GIN classifier hidden dimensions, output channels, learning rate, and batch size.

**`get_train_config(arch, dataset)`** — returns epoch counts for the classifier and generative model per (arch, dataset) combination.

**`get_gen_config(arch)`** — returns default hyperparameters for each generative model (learning rate, latent dim, hidden dim, etc.).

**Defence-specific getters:**

**`get_dp_config(dataset)`** *(config_dp.py only)* — returns `noise_multiplier`, `max_grad_norm`, `delta`, and `use_dp` for each dataset.

**`get_pseudo_edge_config(dataset)`** *(config_ep.py only)* — returns `top_k`, `similarity`, `per_node_topk`, `min_nodes`, and `use_pseudo_edges` for each dataset.

---

## Usage

### Attack Runner (`main.py`)

All attack experiments are launched through `main.py`.

```bash
python main.py --arch <arch> --dataset <dataset> --attack <attack> [options]
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `--arch` | ✅ | Architecture: `proposed` / `random` / `base` |
| `--dataset` | ✅ | One or more dataset names (space-separated) |
| `--attack` | ✅ | Attack type: `logra` / `csgra` / `sota` |
| `--seeds` | ❌ | Random seeds (default: `11 22 33 44 55`) |
| `--epochs-classifier` | ❌ | Override classifier training epochs |
| `--epochs-gen` | ❌ | Override generative model training epochs |
| `--gen-lr` | ❌ | Learning rate for the generative model |
| `--gen-batch-size` | ❌ | Batch size for generative model training |
| `--latent-dim` | ❌ | Latent dimension (GAN / AE) |
| `--hidden-dim` | ❌ | Hidden dimension for generator/encoder/decoder |
| `--lr-g` | ❌ | GAN generator learning rate |
| `--lr-d` | ❌ | GAN discriminator learning rate |
| `--data-root` | ❌ | Root folder for datasets (default: `./data`) |
| `--model-root` | ❌ | Root folder for checkpoints (default: `./models`) |
| `--output-root` | ❌ | Root folder for results (default: `./outputs`) |
| `--skip-classifier` | ❌ | Load classifier from disk instead of training |
| `--skip-gen` | ❌ | Load generative model from disk instead of training |
| `--dry-run` | ❌ | Print resolved config and exit without running |
| `--device` | ❌ | Force device: `cuda` / `cpu` / `mps` |

#### Examples

```bash
# Proposed (Laplace Diffusion) on ENZYMES with seeds 11 and 22
python main.py --arch proposed --dataset ENZYMES --attack logra --seeds 11 22

# GAN on NCI1 with custom epochs
python main.py --arch random --dataset NCI1 --attack logra --epochs-classifier 300 --epochs-gen 70

# CVAE on Cora with all default seeds
python main.py --arch base --dataset Cora --attack csgra

# Multiple datasets in one run
python main.py --arch proposed --dataset Cora ENZYMES NCI1 --attack logra

# Override generative model learning rate and batch size
python main.py --arch proposed --dataset ENZYMES --attack logra --gen-lr 5e-4 --gen-batch-size 16

# Skip classifier training (load saved checkpoint) and run generation only
python main.py --arch base --dataset ENZYMES --attack logra --skip-classifier

# Dry-run: inspect resolved config without running
python main.py --arch proposed --dataset Cora --attack logra --dry-run
```

---

### Defence: Differential Privacy (`main_dp.py`)

Trains the GIN classifier with **DP-SGD** (gradient clipping + Gaussian noise), providing formal (ε, δ)-differential privacy guarantees. The rest of the pipeline (attack, generative model, evaluation) runs unchanged.

```bash
python src/Defence/DifferentialPrivacy/main_dp.py --arch <arch> --dataset <dataset> --attack <attack> [options]
```

#### Additional DP arguments

| Argument | Description |
|---|---|
| `--no-dp` | Disable DP entirely — runs standard SGD (ablation baseline) |
| `--dp-noise σ` | Override `noise_multiplier` for all datasets |
| `--dp-clip C` | Override `max_grad_norm` (gradient clip bound) for all datasets |
| `--dp-delta δ` | Override `delta` for all datasets |

All standard arguments (`--arch`, `--dataset`, `--attack`, `--seeds`, `--epochs-*`, `--gen-*`, `--skip-*`, `--dry-run`, `--device`) are also supported.

#### DP examples

```bash
# DP on ENZYMES, proposed arch, LOGRA attack (default seeds)
python src/Defence/DifferentialPrivacy/main_dp.py \
    --arch proposed --dataset ENZYMES --attack logra

# Multiple datasets in one run
python src/Defence/DifferentialPrivacy/main_dp.py \
    --arch proposed --dataset ENZYMES NCI1 Cora PolBlogs --attack csgra

# Override DP knobs
python src/Defence/DifferentialPrivacy/main_dp.py \
    --arch proposed --dataset NCI1 --attack logra \
    --dp-noise 0.8 --dp-clip 1.5 --dp-delta 1e-5

# Disable DP (standard SGD baseline for comparison)
python src/Defence/DifferentialPrivacy/main_dp.py \
    --arch base --dataset Cora --attack logra --no-dp

# Specific seeds
python src/Defence/DifferentialPrivacy/main_dp.py \
    --arch random --dataset PolBlogs --attack csgra --seeds 11 22

# Dry-run
python src/Defence/DifferentialPrivacy/main_dp.py \
    --arch proposed --dataset ENZYMES --attack logra --dry-run
```

#### How DP-SGD works here

At each training step the defence: (1) clips each parameter's gradient to L2 norm ≤ `max_grad_norm`, bounding the sensitivity; then (2) adds i.i.d. Gaussian noise calibrated to `noise_multiplier × max_grad_norm`, hiding individual contributions. Privacy consumption ε is computed and logged after every epoch using RDP accounting, so you can track the privacy budget as training progresses. Test graphs are never modified.

#### DP output files

Results are tagged with `_dp_` and saved alongside standard outputs:

| File | Description |
|---|---|
| `{dataset}_{arch}_{attack}_dp_seed{seed}.csv` | Distance metrics |
| `structural_{dataset}_{arch}_{attack}_dp_seed{seed}.csv` | Structural metrics |
| `{arch}_{attack}_dp_metrics_summary.txt` | Aggregated mean ± std across all seeds |
| `models/.../train_log_dp.csv` | Epoch log including ε per epoch |

---

### Defence: Edge Perturbation (`main_ep.py`)

Augments **training graphs only** with synthetic pseudo-edges between nodes whose feature vectors are most similar (cosine or Euclidean). This makes the classifier more robust to structural membership-inference attacks without any privacy budget. Test graphs are never modified.

```bash
python src/Defence/EdgePerturbation/main_ep.py --arch <arch> --dataset <dataset> --attack <attack> [options]
```

#### Additional PE arguments

| Argument | Description |
|---|---|
| `--no-pseudo-edges` | Disable augmentation entirely (standard training baseline) |
| `--pe-top-k K` | Number of pseudo-edges added per node (or globally with `--pe-global`) |
| `--pe-similarity` | Similarity metric: `cosine` (default) or `euclidean` |
| `--pe-global` | Use a global top-k edge budget instead of per-node top-k |

All standard arguments are also supported.

#### PE examples

```bash
# PE on ENZYMES, proposed arch, LOGRA attack (default seeds)
python src/Defence/EdgePerturbation/main_ep.py \
    --arch proposed --dataset ENZYMES --attack logra

# Multiple datasets in one run
python src/Defence/EdgePerturbation/main_ep.py \
    --arch proposed --dataset ENZYMES NCI1 Cora PolBlogs --attack csgra

# Override PE knobs
python src/Defence/EdgePerturbation/main_ep.py \
    --arch proposed --dataset Cora --attack logra \
    --pe-top-k 10 --pe-similarity euclidean

# Use global top-k budget instead of per-node
python src/Defence/EdgePerturbation/main_ep.py \
    --arch proposed --dataset ENZYMES --attack logra --pe-global

# Disable pseudo-edges (standard training baseline for comparison)
python src/Defence/EdgePerturbation/main_ep.py \
    --arch base --dataset NCI1 --attack csgra --no-pseudo-edges

# Specific seeds
python src/Defence/EdgePerturbation/main_ep.py \
    --arch random --dataset PolBlogs --attack logra --seeds 11 22

# Dry-run
python src/Defence/EdgePerturbation/main_ep.py \
    --arch proposed --dataset ENZYMES --attack logra --dry-run
```

#### How edge perturbation works here

For each training graph: (1) pairwise similarity between all node feature vectors is computed; (2) the diagonal and existing edges are zeroed out so only genuinely new connections are proposed; (3) for each node (per-node mode) or globally, the top-k most similar unconnected pairs are selected; (4) bidirectional edges are added and the augmented graph replaces the original for training. A summary of edges added is printed at the start of each run so you can verify the augmentation.

#### PE output files

Results are tagged with `_pe_` and saved alongside standard outputs:

| File | Description |
|---|---|
| `{dataset}_{arch}_{attack}_pe_seed{seed}.csv` | Distance metrics |
| `structural_{dataset}_{arch}_{attack}_pe_seed{seed}.csv` | Structural metrics |
| `{arch}_{attack}_pe_metrics_summary.txt` | Aggregated mean ± std across all seeds |
| `models/.../train_log_pe.csv` | Epoch-by-epoch classifier training log |

---

### Comparing attack vs. defences

To compare the undefended attack against both defences on the same dataset and architecture, run all three scripts with matching arguments:

```bash
# Undefended (attack baseline)
python main.py --arch proposed --dataset ENZYMES --attack logra --seeds 11 22

# DP-SGD defence
python src/Defence/DifferentialPrivacy/main_dp.py \
    --arch proposed --dataset ENZYMES --attack logra --seeds 11 22

# Edge Perturbation defence
python src/Defence/EdgePerturbation/main_ep.py \
    --arch proposed --dataset ENZYMES --attack logra --seeds 11 22
```

Results CSVs from all three runs land in the same `outputs/` directory and can be aggregated together for comparison.

---

## Output

After each run, the following files are saved to `--output-root` (default: `./outputs/`):

| File | Description |
|---|---|
| `{dataset}_{arch}_seed{seed}.csv` | Per-graph distance metrics — attack baseline |
| `{dataset}_{arch}_{attack}_dp_seed{seed}.csv` | Per-graph distance metrics — DP defence |
| `{dataset}_{arch}_{attack}_pe_seed{seed}.csv` | Per-graph distance metrics — PE defence |
| `structural_{dataset}_{arch}_seed{seed}.csv` | Structural metrics — attack baseline |
| `structural_...dp_seed{seed}.csv` | Structural metrics — DP defence |
| `structural_...pe_seed{seed}.csv` | Structural metrics — PE defence |
| `{arch}_metrics_summary.txt` | Aggregated mean ± std — attack baseline |
| `{arch}_{attack}_dp_metrics_summary.txt` | Aggregated mean ± std — DP defence |
| `{arch}_{attack}_pe_metrics_summary.txt` | Aggregated mean ± std — PE defence |

Model checkpoints are saved to `--model-root`:

| File | Description |
|---|---|
| `best_model.pth` | Best GIN classifier checkpoint |
| `train_log.csv` | Epoch-by-epoch classifier log (attack baseline) |
| `train_log_dp.csv` | Epoch-by-epoch log including ε per epoch (DP defence) |
| `train_log_pe.csv` | Epoch-by-epoch log (PE defence) |
| `gen_model.pth` | Saved generative model checkpoint |

---

## Evaluation Metrics

### Distance / Embedding Metrics

| Metric | Description |
|---|---|
| **FGD** | Fréchet Graph Distance — embedding-level distributional distance between real and generated graphs |
| **EGD** | L2 distance between GCN embeddings of paired real and generated graphs |
| **MMD** | Maximum Mean Discrepancy between graph embeddings using a Gaussian kernel |
| **GKS** | Weisfeiler-Lehman graph kernel similarity score |
| **SD** | L2 distance between normalised Laplacian eigenvalue spectra |

### Structural Similarity Metrics (Wasserstein-based, reported as %)

| Metric | Description |
|---|---|
| **Degree** | Similarity of degree distributions |
| **LCC** | Similarity of local clustering coefficient distributions |
| **Betweenness** | Similarity of betweenness centrality distributions |
| **Closeness** | Similarity of closeness centrality distributions |

Higher structural scores (closer to 100%) indicate better structural fidelity. Lower distance metrics indicate better generative quality. When comparing defences, higher distance/lower structural scores indicate a more effective defence (the generated graphs diverge more from the real training graphs).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{GraphRevealer2026,
  title  = {LOGRA: Graph Generation Benchmark under Shadow Model Settings},
  author = {Name},
  year   = {2026},
  url    = {}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.