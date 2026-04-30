DATASET_CONFIGS = {
    # ── TUDataset ──────────────────────────────────────────────────────────
    "ENZYMES": {
        "source":        "TUDataset",
        "name":          "ENZYMES",
        "node_features": 3,
        "classes":       6,
        "max_nodes":     126,
        "k_hop":         None,          # not applicable for TUDatasets
    },
    "NCI1": {
        "source":        "TUDataset",
        "name":          "NCI1",
        "node_features": 37,
        "classes":       2,
        "max_nodes":     126,
        "k_hop":         None,
    },
    # ── Node → Graph (ego subgraph) datasets ──────────────────────────────
    "Cora": {
        "source":        "Planetoid",
        "name":          "Cora",
        "node_features": -1,            # inferred at runtime
        "classes":       7,
        "max_nodes":     500,
        "k_hop":         4,
    },
    "USA": {
        "source":        "Airports",
        "name":          "USA",
        "node_features": -1,            # inferred at runtime
        "classes":       4,
        "max_nodes":     100,
        "k_hop":         4,
    },
    # ── PolBlogs (kept for completeness) ──────────────────────────────────
    "PolBlogs": {
        "source":        "PolBlogs",
        "name":          "PolBlogs",
        "node_features": -1,
        "classes":       2,
        "max_nodes":     500,
        "k_hop":         2,
    },
}

# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------
# Maps the CLI --arch flag to a human-readable description and internal key.

ARCH_CONFIGS = {
    # ── proposed: Laplace Diffusion model (LOGRA proposed) ────────────────
    "proposed": {
        "description":   "Laplace Diffusion graph generation model",
        "gen_model":     "diffusion",
    },
    # ── random: Conditional GAN graph generation model ────────────────────
    "random": {
        "description":   "Conditional GAN graph generation model",
        "gen_model":     "gan",
    },
    # ── base: Conditional VAE (CVAE) graph generation model ───────────────
    "base": {
        "description":   "Conditional VAE (CVAE) graph generation model",
        "gen_model":     "cvae",
    },
}

ATTACK_CONFIGS = {
    # ── logra: LOGRA attack ────────────────────────────────────────────────
    "logra": {
        "description":   "LOGRA attack",
        
    },
    # ── csgra: CSGRA attack ────────────────────────────────────────────────
    "csgra": {
        "description":   "CSGRA attack",
     
    },
    # ── sota: State-of-the-art graph generation model ───────────────────────
    "sota": {
        "description":   "SOTA attack",
      
    },
}

# ---------------------------------------------------------------------------
# Training hyperparameters (per architecture × dataset)
# ---------------------------------------------------------------------------
# Keys are (arch, dataset_name). Fall back to ("default", "default") when
# an exact match is not found.

TRAIN_CONFIGS = {
    # ── proposed (diffusion) ──────────────────────────────────────────────
    ("proposed", "ENZYMES"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },
    ("proposed", "NCI1"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },
    ("proposed", "Cora"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },
  

    # ── random (GAN) ──────────────────────────────────────────────────────
    ("random", "ENZYMES"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },
    ("random", "NCI1"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },

    # ── base (CVAE) ───────────────────────────────────────────────────────
    ("base", "ENZYMES"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },
    ("base", "NCI1"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },
    ("base", "Cora"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },
    ("base", "USA"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },

    # ── fallback defaults ─────────────────────────────────────────────────
    ("default", "default"): {
        "num_epoch_classifier": 300,
        "num_epoch_gen":        70,
    },
}

# ---------------------------------------------------------------------------
# GIN classifier hyperparameters (shared across all arches)
# ---------------------------------------------------------------------------
GIN_CONFIG = {
    "hidden_channels": 1000,
    "out_channels":    100,
    "lr":              0.001,
    "batch_size":      64,
}

# ---------------------------------------------------------------------------
# Generative model hyperparameters
# ---------------------------------------------------------------------------
DIFFUSION_CONFIG = {
    "batch_size":    32,
    "lr":            1e-3,
    "num_timesteps": 100,
    "epsilon":       0.5,
    "latent_dim":    128,       # unused by diffusion but kept for symmetry
    "hidden_dim":    256,
}

GAN_CONFIG = {
    "batch_size":  32,
    "lr_g":        2e-4,
    "lr_d":        2e-4,
    "latent_dim":  128,
    "hidden_dim":  256,
}

CVAE_CONFIG = {
    "batch_size":  32,
    "lr":          2e-3,
    "latent_dim":  128,
    "hidden_dim":  256,
    "beta_schedule": "linear",   # "constant" | "linear" | "cyclical"
}

# ---------------------------------------------------------------------------
# Graph-distance filter (NCI1GraphDataset)
# ---------------------------------------------------------------------------
GRAPH_FILTER = {
    "min_nodes":   2,       # proposed/base use 2; random uses 8 (harder filter)
    "max_degree":  5.0,
}

# ---------------------------------------------------------------------------
# Seeds used in the paper experiments
# ---------------------------------------------------------------------------
DEFAULT_SEEDS = [11, 22, 33, 44, 55]

# ---------------------------------------------------------------------------
# Output / paths
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT   = "data"
DEFAULT_MODEL_ROOT  = "artifacts"
DEFAULT_OUTPUT_ROOT = "outputs"

RESULT_DIR = "results"
# ---------------------------------------------------------------------------
# Convenience helper used by main.py
# ---------------------------------------------------------------------------

def get_train_config(arch: str, dataset_name: str) -> dict:
    """Return training-epoch config for the given (arch, dataset) pair."""
    key = (arch, dataset_name)
    if key in TRAIN_CONFIGS:
        return dict(TRAIN_CONFIGS[key])
    return dict(TRAIN_CONFIGS[("default", "default")])


def get_gen_config(arch: str) -> dict:
    """Return the generative-model hyperparameter dict for *arch*."""
    mapping = {
        "proposed": DIFFUSION_CONFIG,
        "random":   GAN_CONFIG,
        "base":     CVAE_CONFIG,
    }
    return dict(mapping.get(arch, DIFFUSION_CONFIG))