"""
configs/rcan_config.py
======================
Central configuration file for the RCAN training pipeline.

All hyperparameters live here. You should NOT need to touch any other file
to change training settings — just edit this file and re-run train.py.

Phase-1 defaults are set for a small dataset (BSD500 subset) so you can
verify the pipeline quickly. To switch to DIV2K for production training,
only the DATASET section needs changing.
"""

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
# Supported: "bsd500" (small, auto-downloads) | "div2k" (production)
DATASET_NAME   = "bsd500"

# Root directory where the dataset lives (or will be downloaded to)
DATA_ROOT      = "./data/downloads"

# Scale factor for super-resolution.
# 2 → double the resolution (most common for benchmarks)
# 4 → quadruple (used in medical imaging scenarios)
SCALE_FACTOR   = 2

# Size of the LR patch sampled during training.
# HR patch size = LR_PATCH_SIZE * SCALE_FACTOR
# Larger = more context, but slower and needs more VRAM.
LR_PATCH_SIZE  = 48   # → 96×96 HR patches at ×2

# Noise level range during degradation (sigma, in [0, 255] scale).
# RCAN is trained to handle a *range* of noise levels, not a fixed one.
# This is the "Microscope" characteristic: robust to varying noise.
NOISE_SIGMA_MIN = 0
NOISE_SIGMA_MAX = 50

# ---------------------------------------------------------------------------
# MODEL ARCHITECTURE (RCAN paper defaults)
# ---------------------------------------------------------------------------
# Number of Residual Groups in the Residual-in-Residual structure.
# Paper uses 10; reduce to 5 for faster experiments.
N_RESGROUPS    = 10

# Number of Residual Channel Attention Blocks per Residual Group.
# Paper uses 20; reduce to 10 for faster experiments.
N_RESBLOCKS    = 20

# Number of feature channels throughout the network.
# Paper uses 64. Increasing improves quality but costs VRAM quadratically.
N_FEATS        = 64

# Reduction ratio for the Channel Attention bottleneck FC layers.
# Channels go: N_FEATS → N_FEATS//REDUCTION → N_FEATS
# Paper uses 16. Lower = more parameters in CA, higher = more compression.
REDUCTION      = 16

# Residual scaling factor. Multiplies residual branch output before adding
# to skip connection. Prevents training instability in very deep networks.
# Paper uses 1 (no scaling). 0.1 can help for very deep configurations.
RES_SCALE      = 1.0

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
BATCH_SIZE     = 16           # Reduce to 4 if you hit VRAM limits
NUM_EPOCHS     = 100          # Paper trains for ~1000; 100 is enough to see convergence
LEARNING_RATE  = 1e-4         # Adam initial LR (paper: 1e-4, halved every 200 epochs)
LR_DECAY_STEP  = 200          # Halve LR every N epochs
LR_DECAY_GAMMA = 0.5          # Multiplicative factor: new_lr = lr * gamma

# Adam optimizer betas (paper default)
ADAM_BETA1     = 0.9
ADAM_BETA2     = 0.999

# Loss function weights:
#   L1 loss: pixel-level accuracy (primary)
#   Perceptual loss: feature-level quality (optional, set to 0 to disable)
LOSS_L1_WEIGHT          = 1.0
LOSS_PERCEPTUAL_WEIGHT  = 0.0   # Set to e.g. 0.01 to enable perceptual loss

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
NUM_WORKERS    = 4            # CPU workers for DataLoader; set to 0 on Windows
PIN_MEMORY     = True         # Speed up GPU transfers; set to False if CPU-only

# Data augmentation (applied to training patches only):
AUG_HFLIP      = True         # Random horizontal flip
AUG_VFLIP      = True         # Random vertical flip
AUG_ROT90      = True         # Random 90° rotation

# ---------------------------------------------------------------------------
# OUTPUT / CHECKPOINTING
# ---------------------------------------------------------------------------
OUTPUT_DIR          = "./outputs"
CHECKPOINT_DIR      = "./outputs/checkpoints"
LOG_DIR             = "./outputs/logs"
SAMPLE_DIR          = "./outputs/samples"

# Save a checkpoint every N epochs
SAVE_EVERY          = 10

# Keep only the N best checkpoints (by PSNR) to save disk space
MAX_KEEP_CHECKPOINTS = 3

# Log training metrics to CSV every N batches
LOG_EVERY_N_BATCHES = 50

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
# Number of samples to save as visual comparison images during eval
NUM_VISUAL_SAMPLES  = 8

# Benchmark comparison: also compute bicubic upscaling baseline
COMPUTE_BICUBIC_BASELINE = True

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
SEED = 42
