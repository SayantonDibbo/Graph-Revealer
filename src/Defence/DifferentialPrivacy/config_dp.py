# config_dp.py
"""
LOGRA – Differential Privacy (DP-SGD) Defence Configuration
============================================================
Centralised hyperparameters for main_dp.py.

Sections
--------
1. Paths & seeds
2. GIN classifier config
3. Dataset configs
4. Architecture configs
5. Attack configs
6. Generative model hyperparameters
7. Training epoch table
8. DP hyperparameters (per dataset)
9. Public getter functions:
     get_train_config(arch, dataset_name)
     get_gen_config(arch)
     get_dp_config(dataset_name)
"""

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
# Keys used by main_dp.py:
#   classes        – number of graph-level classes
#   node_features  – input feature dim (-1 = infer from data at runtime)
#   max_nodes      – maximum nodes across graphs (used to pad tensors)
#   k_hop          – ego-subgraph radius (PolBlogs only)

DATASET_CONFIGS = {
    "ENZYMES": {
        "classes":       6,
        "node_features": 18,    # standard ENZYMES node-label one-hot dim
        "max_nodes":     126,
        "k_hop":         2,
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
# 'description' is printed at start-up (used in main() startup banner).

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
        "lr":      2e-4,    # shared fallback (overridden by lr_g / lr_d)
        "lr_g":    2e-4,
        "lr_d":    2e-4,
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
# Cols  : per-dataset overrides; fallback to "default" if dataset not listed.
#
# DP training typically needs fewer epochs than standard training because
# noisy gradients slow convergence — tune these per your hardware budget.

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
# 8. DP hyperparameters (per dataset)
# ===========================================================================
# These are the baseline defaults consumed by get_dp_config().
# CLI flags (--dp-noise, --dp-clip, --dp-delta, --no-dp) are applied on top
# of these values inside run_experiment() before constructing PrivacyConfig.
#
# Tuning guidance:
#   noise_multiplier – higher σ → stronger privacy, noisier gradients.
#                      Typical range: 0.8 – 2.0.
#   max_grad_norm    – per-parameter gradient clip bound C.
#                      Too small clips real signal; too large weakens privacy.
#                      Typical range: 0.5 – 2.0.
#   delta            – should be << 1/|dataset|.
#                      1e-5 is a safe default for datasets of ≥ 1 000 samples.

_DP_CONFIGS = {
    "ENZYMES": {
        "noise_multiplier": 1.1,
        "max_grad_norm":    1.0,
        "delta":            1e-5,
        "use_dp":           True,
    },
    "NCI1": {
        "noise_multiplier": 1.1,
        "max_grad_norm":    1.0,
        "delta":            1e-5,
        "use_dp":           True,
    },
    "Cora": {
        "noise_multiplier": 1.0,
        "max_grad_norm":    1.0,
        "delta":            1e-5,
        "use_dp":           True,
    },
    "PolBlogs": {
        "noise_multiplier": 1.1,
        "max_grad_norm":    1.0,
        "delta":            1e-5,
        "use_dp":           True,
    },
}

# Fallback used when a dataset is not listed above
_DEFAULT_DP_CONFIG = {
    "noise_multiplier": 1.1,
    "max_grad_norm":    1.0,
    "delta":            1e-5,
    "use_dp":           True,
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


def get_dp_config(dataset_name: str) -> dict:
    """
    Return a *copy* of the DP hyperparameter dict for *dataset_name*.

    Falls back to _DEFAULT_DP_CONFIG if the dataset is not listed.
    The returned dict is what main_dp.py mutates with CLI overrides before
    constructing a PrivacyConfig dataclass.
    """
    base = _DP_CONFIGS.get(dataset_name, _DEFAULT_DP_CONFIG)
    return dict(base)