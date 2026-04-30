# config_ep.py


# ===========================================================================
# 1. Paths & seeds
# ===========================================================================

DEFAULT_DATA_ROOT   = "data"
DEFAULT_MODEL_ROOT  = "models"
DEFAULT_OUTPUT_ROOT = "results"

DEFAULT_SEEDS = [42, 123, 456]


# ===========================================================================
# 2. GIN classifier config
# ===========================================================================

GIN_CONFIG = {
    "hidden_channels": 64,
    "out_channels":    64,
    "lr":              1e-3,
    "batch_size":      32,
}


# ===========================================================================
# 3. Dataset configs
# ===========================================================================
# Keys used by main_pe.py:
#   classes        – number of graph-level classes
#   node_features  – input feature dim (-1 = infer from data)
#   max_nodes      – maximum nodes across graphs (used to pad tensors)
#   k_hop          – ego-subgraph radius (PolBlogs only)

DATASET_CONFIGS = {
    "ENZYMES": {
        "classes":       6,
        "node_features": 18,    # standard ENZYMES node-label one-hot dim
        "max_nodes":     126,
        "k_hop":         2,     # unused for ENZYMES but kept for uniformity
    },
    "NCI1": {
        "classes":       2,
        "node_features": 37,    # NCI1 discrete node-label one-hot dim
        "max_nodes":     111,
        "k_hop":         2,
    },
    "Cora": {
        "classes":       7,
        "node_features": -1,    # inferred at runtime from data
        "max_nodes":     200,
        "k_hop":         2,
    },
    "PolBlogs": {
        "classes":       2,
        # node_features is set at runtime to num_classes (one-hot of labels)
        "node_features": 2,
        "max_nodes":     500,
        "k_hop":         2,
    },
}


# ===========================================================================
# 4. Architecture configs
# ===========================================================================
# 'description' is printed at start-up for human readability.
# Remaining keys are available via ARCH_CONFIGS[arch] but generative
# hyperparameters are accessed through get_gen_config() below.

ARCH_CONFIGS = {
    "proposed": {
        "description": "Graph Diffusion Model",
    },
    "random": {
        "description": "Graph GAN",
    },
    "base": {
        "description": "Graph CVAE",
    },
}


# ===========================================================================
# 5. Attack configs
# ===========================================================================

ATTACK_CONFIGS = {
    "logra":  {"description": "LOGRA membership-inference attack"},
    "csgra":  {"description": "CSGRA membership-inference attack"},
}


# ===========================================================================
# 6. Generative model hyperparameters
# ===========================================================================

_GEN_CONFIGS = {
    # ── Diffusion (proposed) ─────────────────────────────────────────────────
    "proposed": {
        "lr":             1e-3,
        "batch_size":     32,
        "hidden_dim":     256,
        "latent_dim":     128,   # not used by diffusion but kept for CLI compat
        "num_timesteps":  100,
        "epsilon":        0.5,
        "beta_schedule":  "linear",   # "constant" | "linear" | "cyclical"
    },
    # ── GAN (random) ─────────────────────────────────────────────────────────
    "random": {
        "lr":          2e-4,    # shared fallback (overridden by lr_g / lr_d)
        "lr_g":        2e-4,
        "lr_d":        2e-4,
        "batch_size":  32,
        "hidden_dim":  256,
        "latent_dim":  128,
    },
    # ── CVAE (base) ──────────────────────────────────────────────────────────
    "base": {
        "lr":            2e-3,
        "batch_size":    32,
        "hidden_dim":    256,
        "latent_dim":    128,
        "beta_schedule": "linear",   # "constant" | "linear" | "cyclical"
    },
}


# ===========================================================================
# 7. Training epoch table
# ===========================================================================
# Rows  : generative architectures
# Cols  : per-dataset overrides; fallback to "default" if dataset not listed

_TRAIN_CONFIGS = {
    "proposed": {
        "default":  {"num_epoch_classifier": 100, "num_epoch_gen": 200},
        "ENZYMES":  {"num_epoch_classifier": 100, "num_epoch_gen": 200},
        "NCI1":     {"num_epoch_classifier": 100, "num_epoch_gen": 200},
        "Cora":     {"num_epoch_classifier": 100, "num_epoch_gen": 150},
        "PolBlogs": {"num_epoch_classifier": 100, "num_epoch_gen": 150},
    },
    "random": {
        "default":  {"num_epoch_classifier": 100, "num_epoch_gen": 300},
        "ENZYMES":  {"num_epoch_classifier": 100, "num_epoch_gen": 300},
        "NCI1":     {"num_epoch_classifier": 100, "num_epoch_gen": 300},
        "Cora":     {"num_epoch_classifier": 100, "num_epoch_gen": 200},
        "PolBlogs": {"num_epoch_classifier": 100, "num_epoch_gen": 200},
    },
    "base": {
        "default":  {"num_epoch_classifier": 100, "num_epoch_gen": 200},
        "ENZYMES":  {"num_epoch_classifier": 100, "num_epoch_gen": 200},
        "NCI1":     {"num_epoch_classifier": 100, "num_epoch_gen": 200},
        "Cora":     {"num_epoch_classifier": 100, "num_epoch_gen": 150},
        "PolBlogs": {"num_epoch_classifier": 100, "num_epoch_gen": 150},
    },
}


# ===========================================================================
# 8. Pseudo-edge augmentation defaults (per dataset)
# ===========================================================================
# These are the baseline defaults consumed by get_pseudo_edge_config().
# CLI flags (--pe-top-k, --pe-similarity, --pe-global, --no-pseudo-edges)
# are applied on top of these values inside run_experiment().

_PSEUDO_EDGE_CONFIGS = {
    "ENZYMES": {
        "top_k":            5,
        "similarity":       "cosine",
        "per_node_topk":    True,
        "min_nodes":        4,
        "use_pseudo_edges": True,
    },
    "NCI1": {
        "top_k":            5,
        "similarity":       "cosine",
        "per_node_topk":    True,
        "min_nodes":        4,
        "use_pseudo_edges": True,
    },
    "Cora": {
        "top_k":            8,
        "similarity":       "cosine",
        "per_node_topk":    True,
        "min_nodes":        4,
        "use_pseudo_edges": True,
    },
    "PolBlogs": {
        "top_k":            5,
        "similarity":       "euclidean",
        "per_node_topk":    True,
        "min_nodes":        4,
        "use_pseudo_edges": True,
    },
}

# Fallback used when a dataset is not listed above
_DEFAULT_PSEUDO_EDGE_CONFIG = {
    "top_k":            5,
    "similarity":       "cosine",
    "per_node_topk":    True,
    "min_nodes":        4,
    "use_pseudo_edges": True,
}


# ===========================================================================
# 9. Public getter functions
# ===========================================================================

def get_train_config(arch: str, dataset_name: str) -> dict:
    """
    Return the training-epoch config for a given (arch, dataset) pair.

    Falls back to the arch's 'default' entry if the dataset has no explicit
    override.  Returns a copy so callers cannot mutate the table.
    """
    arch_table = _TRAIN_CONFIGS.get(arch)
    if arch_table is None:
        raise ValueError(
            f"Unknown arch {arch!r}. "
            f"Choose from: {list(_TRAIN_CONFIGS.keys())}"
        )
    row = arch_table.get(dataset_name, arch_table["default"])
    return dict(row)


def get_gen_config(arch: str) -> dict:
    """
    Return a *copy* of the generative model hyperparameter dict for *arch*.

    Returning a copy means CLI overrides applied in resolve_config() do not
    permanently modify the module-level table.
    """
    cfg = _GEN_CONFIGS.get(arch)
    if cfg is None:
        raise ValueError(
            f"Unknown arch {arch!r}. "
            f"Choose from: {list(_GEN_CONFIGS.keys())}"
        )
    return dict(cfg)


def get_pseudo_edge_config(dataset_name: str) -> dict:
    """
    Return a *copy* of the pseudo-edge hyperparameter dict for *dataset_name*.

    Falls back to _DEFAULT_PSEUDO_EDGE_CONFIG if the dataset is not listed.
    The returned dict is what main_pe.py mutates with CLI overrides before
    constructing a PseudoEdgeConfig dataclass.
    """
    base = _PSEUDO_EDGE_CONFIGS.get(dataset_name, _DEFAULT_PSEUDO_EDGE_CONFIG)
    return dict(base)