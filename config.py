import torch

SIGNAL_LENGTH = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
LATENT_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Noise parameters
GAUSSIAN_SIGMA = 0.1
MASK_PROB = 0.1
MASK_LEN = 20

# Experiment ranges
LATENT_DIM_SWEEP = [8, 16, 32, 64, 128, 256]
NOISE_SIGMA_SWEEP = [0.05, 0.1, 0.2, 0.4]
LR_SWEEP = [1e-4, 5e-4, 1e-3]
BATCH_SWEEP = [32, 64, 128]

# Paths
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
