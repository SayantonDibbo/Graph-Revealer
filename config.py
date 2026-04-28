# config.py

# ======================
# GENERAL SETTINGS
# ======================
SEEDS = [11, 22, 33, 44, 55]
DEFAULT_SEED = 11

DATASET_NAME = "Cora"
ROOT_DIR = "data"
K_HOP = 2

# ======================
# TRAINING CONFIG
# ======================
BATCH_SIZE = 64
TEST_SIZE = 0.2
SHADOW_SPLIT = 0.5
RANDOM_STATE = 42

# ======================
# MODEL CONFIG (GIN)
# ======================
HIDDEN_CHANNELS = 1000
OUT_CHANNELS = 100
LEARNING_RATE = 0.001

# ======================
# DATASET-SPECIFIC CONFIG
# ======================
DATASET_CONFIG = {
    "Cora": {
        "num_epoch1": 70,
        "num_epoch2": 70,
        "classes": 7,
        "max_node": 200,
    },
    "USA": {
        "num_epoch1": 50,
        "num_epoch2": 50,
        "classes": 4,
        "max_node": 100,
    }
}

# ======================
# DIFFUSION CONFIG
# ======================
DIFFUSION = {
    "num_timesteps": 80,
    "epsilon": 0.5,
    "batch_size": 32,
    "epochs": 150,
    "lr": 2e-3
}

# ======================
# PATHS
# ======================
MODEL_DIR = "models"
FIGURE_DIR = "figures"