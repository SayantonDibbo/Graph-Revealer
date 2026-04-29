# =============================================================================
# config.py  –  All constants for the LOGRA pipeline
#               Covers both "proposed" (diffusion) and "baseline" (CVAE) modes
# =============================================================================

# ── Reproducibility ───────────────────────────────────────────────────────────
SEEDS = [11, 22, 33, 44, 66]

# ── Experiment modes ──────────────────────────────────────────────────────────
# Pass via --experiment-type at the CLI.
EXPERIMENT_PROPOSED = "proposed"   # Laplace-diffusion generator
EXPERIMENT_BASELINE = "baseline"   # CVAE generator
EXPERIMENT_TYPE     = EXPERIMENT_PROPOSED   # default

# ── Supported dataset names ───────────────────────────────────────────────────
SUPPORTED_DATASETS = ["Cora", "USA"]

# ── Dataset-specific hyperparameters ─────────────────────────────────────────
# Each entry carries epoch counts for BOTH the GIN classifier (num_epoch1)
# and the generative model (num_epoch2), plus the per-mode MAX_NODE values.
# The baseline uses larger ego-graphs: k_hop=3 and bigger MAX_NODE.
DATASET_CONFIG = {
    "Cora": {
        "dataset_name"        : "Cora",
        "classes"             : 7,
        # proposed (diffusion) settings
        "proposed_MAX_NODE"   : 200,
        "proposed_num_epoch1" : 70,
        "proposed_num_epoch2" : 70,
        "proposed_k_hop"      : 2,
        # baseline (CVAE) settings
        "baseline_MAX_NODE"   : 500,
        "baseline_num_epoch1" : 50,
        "baseline_num_epoch2" : 50,
        "baseline_k_hop"      : 3,
    },
    "USA": {
        "dataset_name"        : "USA",
        "classes"             : 4,
        # proposed
        "proposed_MAX_NODE"   : 100,
        "proposed_num_epoch1" : 50,
        "proposed_num_epoch2" : 50,
        "proposed_k_hop"      : 2,
        # baseline
        "baseline_MAX_NODE"   : 100,
        "baseline_num_epoch1" : 20,
        "baseline_num_epoch2" : 20,
        "baseline_k_hop"      : 3,
    },
}

# ── Dataset prefixes used when globbing result CSVs ──────────────────────────
# proposed outputs:  Cor_proposed_seed*.csv  → first 3 chars = "Cor"
# baseline outputs:  Cor_baseline_seed*.csv  → first 3 chars = "Cor"
DATASET_PREFIXES_PROPOSED = ["Cor"]
DATASET_PREFIXES_BASELINE = ["Cor"]

# ── Data root ─────────────────────────────────────────────────────────────────
DATA_ROOT = "data"

# ── Train / test / shadow split ratios ───────────────────────────────────────
SHADOW_SPLIT_RATIO  = 0.5
TEST_SPLIT_RATIO    = 0.2
SPLIT_RANDOM_STATE  = 42

# =============================================================================
# GIN Classifier  (shared by both modes)
# =============================================================================
GIN_HIDDEN_CHANNELS = 1000
GIN_OUT_CHANNELS    = 100
GIN_LR              = 0.001
GIN_BATCH_SIZE      = 64

# =============================================================================
# NCI1GraphDataset filters
# (MIN_NODES differs: proposed keeps ≥ 2, baseline keeps ≥ 8)
# =============================================================================
MAX_AVG_DEGREE      = 5.0
PROPOSED_MIN_NODES  = 2
BASELINE_MIN_NODES  = 8

# =============================================================================
# Proposed mode – Laplace Diffusion
# =============================================================================

# ── Model ─────────────────────────────────────────────────────────────────────
DIFF_BATCH_SIZE          = 32
DIFF_LR                  = 2e-3
DIFF_NUM_TIMESTEPS       = 80
DIFF_EPSILON             = 0.5

# Beta schedule
BETA_START               = 1e-4
BETA_END                 = 0.02

# Noise clamping
LAPLACE_CLAMP_MIN        = -5.0
LAPLACE_CLAMP_MAX        =  5.0

# ── GraphDiffusionNetwork ─────────────────────────────────────────────────────
GDN_HIDDEN_DIM           = 256
GDN_TIME_DIM             = 128
GDN_DROPOUT              = 0.1
GDN_TIME_SCALE           = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
DIFF_WEIGHT_DECAY        = 1e-4
DIFF_ADAM_BETA1          = 0.9
DIFF_ADAM_BETA2          = 0.999
DIFF_GRAD_CLIP           = 0.5
DIFF_WARMUP_EPOCHS       = 10
FEAT_LOSS_WEIGHT         = 2.0   # total = adj_loss + FEAT_LOSS_WEIGHT * feat_loss

# ── Sampling (reverse diffusion) ─────────────────────────────────────────────
SAMPLE_MIN_NODES         = 15
SAMPLE_MAX_NODES         = 25
INITIAL_NOISE_SCALE      = 0.5
POSTERIOR_NOISE_SCALE    = 0.5
ADJ_CLAMP_MIN            = -1.0
ADJ_CLAMP_MAX            =  1.0
FEAT_CLAMP_MIN           = -3.0
FEAT_CLAMP_MAX           =  3.0
ADJ_THRESHOLD_PERCENTILE = 0.70

# =============================================================================
# Baseline mode – Conditional VAE (CVAE)
# =============================================================================

# ── Architecture ─────────────────────────────────────────────────────────────
CVAE_HIDDEN_DIM          = 256
CVAE_LATENT_DIM          = 128
CVAE_ENC_ADJ_DROP        = 0.4
CVAE_ENC_FEAT_DROP       = 0.2
CVAE_ENC_JOINT_DROP      = 0.2
CVAE_DEC_DROP            = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
CVAE_BATCH_SIZE          = 32
CVAE_LR                  = 2e-3
CVAE_WEIGHT_DECAY        = 1e-4
CVAE_GRAD_CLIP           = 1.0
CVAE_BETA_SCHEDULE       = "linear"   # "constant" | "linear" | "cyclical"
CVAE_BETA_MAX            = 1.0
CVAE_SPARSITY_WEIGHT     = 0.05

# ── Laplace reparameterisation ────────────────────────────────────────────────
CVAE_LAPLACE_EPSILON     = 0.5
CVAE_LAPLACE_CLAMP       = 10.0      # symmetric clamp on noise samples

# ── Sampling ─────────────────────────────────────────────────────────────────
CVAE_ADJ_THRESHOLD       = 0.5
CVAE_SAMPLE_MIN_NODES    = 15
CVAE_SAMPLE_MAX_NODES    = 25

# ── Output paths ─────────────────────────────────────────────────────────────
CVAE_SAVE_DIR            = "LOGRA_base"
CVAE_MODEL_FILENAME      = "cvae_model.pth"

# =============================================================================
# Shared metrics & evaluation
# =============================================================================
ENCODER_HIDDEN_DIM       = 256
ENCODER_EMBED_DIM        = 128
ENCODER_EPS              = 1e-6
SPECTRAL_K               = 20
WL_ITERATIONS            = 2
GC_FLUSH_EVERY           = 5

# =============================================================================
# Shared output / model paths
# =============================================================================
MODELS_DIR               = "models"
FIGURES_DIR              = "figures"
BEST_MODEL_FILENAME      = "best_model.pth"
TRAIN_LOG_FILENAME       = "train_log.csv"
RESULT_DIR               = "results"
# =============================================================================
# Shared visualisation constants
# =============================================================================
VIZ_NUM_SAMPLES          = 5
VIZ_GRID_ROWS            = 4
VIZ_GRID_COLS            = 5
VIZ_NODE_SIZE_SMALL      = 50
VIZ_NODE_SIZE_MED        = 140
VIZ_NODE_SIZE_LARGE      = 450
VIZ_DPI                  = 150
VIZ_COLORMAP             = "tab10"
VIZ_EDGE_CMAP            = "coolwarm"
VIZ_FEAT_VMIN            = -2.0
VIZ_FEAT_VMAX            =  2.0
VIZ_SPRING_K             = 1.5
VIZ_SPRING_ITER          = 150