import os

# Directories
DATA_DIR   = "results/data"
PLOT_DIR   = "results/plots"
MODEL_DIR  = "results/models"

# Signal segmentation
WINDOW_SIZE = 1024
STRIDE      = 512
SAMPLE_RATE = 12000  # Hz

# Dataset balancing (simulate factory: mostly normal)
FAULT_RATIO = 0.10   # keep 10 % of fault segments

# VAE hyper-parameters
LATENT_DIM = 32
BETA       = 1.0     # KL weight (beta-VAE; 1.0 = standard VAE)

# Training
BATCH_SIZE    = 64
N_EPOCHS      = 100
LEARNING_RATE = 1e-3
K_FOLDS       = 5
PATIENCE      = 20   # early stopping patience

# Anomaly threshold: percentile of normal reconstruction errors
THRESHOLD_PERCENTILE = 95

# ── CWRU download map ────────────────────────────────────────────────────────
BASE_URL = "https://engineering.case.edu/sites/default/files/"

# Each entry: (filename, label_name, label_id)
# label_id: 0=Normal, 1=InnerRace, 2=Ball, 3=OuterRace
FILE_CONFIGS = [
    # Normal baseline (0-3 HP)
    ("97.mat",   "Normal",   0),
    ("98.mat",   "Normal",   0),
    ("99.mat",   "Normal",   0),
    ("100.mat",  "Normal",   0),
    # Inner Race 0.007"
    ("105.mat",  "IR_007",   1),
    ("106.mat",  "IR_007",   1),
    ("107.mat",  "IR_007",   1),
    ("108.mat",  "IR_007",   1),
    # Inner Race 0.014"
    ("169.mat",  "IR_014",   1),
    ("170.mat",  "IR_014",   1),
    ("171.mat",  "IR_014",   1),
    ("172.mat",  "IR_014",   1),
    # Inner Race 0.021"
    ("209.mat",  "IR_021",   1),
    ("210.mat",  "IR_021",   1),
    ("211.mat",  "IR_021",   1),
    ("212.mat",  "IR_021",   1),
    # Ball 0.007"
    ("118.mat",  "Ball_007", 2),
    ("119.mat",  "Ball_007", 2),
    ("120.mat",  "Ball_007", 2),
    ("121.mat",  "Ball_007", 2),
    # Ball 0.014"
    ("185.mat",  "Ball_014", 2),
    ("186.mat",  "Ball_014", 2),
    ("187.mat",  "Ball_014", 2),
    ("188.mat",  "Ball_014", 2),
    # Ball 0.021"
    ("222.mat",  "Ball_021", 2),
    ("223.mat",  "Ball_021", 2),
    ("224.mat",  "Ball_021", 2),
    ("225.mat",  "Ball_021", 2),
    # Outer Race @6 0.007"
    ("130.mat",  "OR_007",   3),
    ("131.mat",  "OR_007",   3),
    ("132.mat",  "OR_007",   3),
    ("133.mat",  "OR_007",   3),
    # Outer Race @6 0.014"
    ("197.mat",  "OR_014",   3),
    ("198.mat",  "OR_014",   3),
    ("199.mat",  "OR_014",   3),
    ("200.mat",  "OR_014",   3),
    # Outer Race @6 0.021"
    ("234.mat",  "OR_021",   3),
    ("235.mat",  "OR_021",   3),
    ("236.mat",  "OR_021",   3),
    ("237.mat",  "OR_021",   3),
]

LABEL_NAMES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Ball Fault",
    3: "Outer Race Fault",
}

COLORS = {
    0: "#2196F3",   # blue
    1: "#F44336",   # red
    2: "#FF9800",   # orange
    3: "#9C27B0",   # purple
}
