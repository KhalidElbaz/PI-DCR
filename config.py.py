import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data parameters
IMAGE_SIZE = 1024
BATCH_SIZE = 2
EPOCHS = 200
LEARNING_RATE = 1e-3  # Changed from 1e-4 as per Table 2

# Material parameters (from paper)
E = 35e9  # Young's modulus (Pa)
nu = 0.25  # Poisson's ratio
Gc = 1.0  # Fracture toughness (from paper)

# Phase field length scale
lc = 0.01

# Loss weights - OPTIMAL CONFIGURATION FROM FIG. 22
w1 = 0.26  # Elasticity loss weight
w2 = 0.38  # Phase field loss weight
w3 = 0.19  # Boundary condition loss weight
w4 = 0.17  # Data loss weight

# MC Dropout parameters (from paper)
dropout_probability = 0.3
MC_forward_passes = 50

# Training parameters
TRAIN_SPLIT = 0.8
NUM_FOLDS = 5
PATIENCE = 50  # Early stopping