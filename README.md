[English](README.md) | [日本語](README.ja.md)

# IL: Imitation Learning Algorithms Library

A unified library implementing four imitation learning algorithms for robotic manipulation with the [Hiwonder ArmPi Pro](https://www.hiwonder.com/). Provides a consistent interface for training and inference across different algorithm architectures.

## Implemented Algorithms

| Algorithm | Description | Key Architecture |
|-----------|-------------|------------------|
| **ACT** (Action Chunking with Transformers) | Transformer-based action sequence prediction with VAE for multi-modal behavior | DETR + ResNet-18 + VAE |
| **Diffusion Policy** | Denoising diffusion model for policy generation | DDPM/DDIM + CNN/Transformer backbone |
| **MLP** (Baseline) | Simple feedforward network for single-step prediction | ResNet-18 + MLP fusion head |
| **LPIL** (Latent Policy Imitation Learning) | Latent representation learning with goal conditioning | External LPIL framework |

## Architecture

```
┌─────────────────────────────────────────────┐
│         User Interface                       │
│   train_main.py (training)                   │
│   model_load.py (inference, factory pattern) │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
   Training          Inference
       │                │
   ┌───┴───┬────┬──────┐
   ▼       ▼    ▼      ▼
  MLP    ACT  Diff   LPIL
   └───┬───┴────┴──────┘
       ▼
  common/
  ├── base_model.py    # Abstract base class for inference
  ├── trainer.py       # Base trainer with early stopping & checkpointing
  ├── armpi_const.py   # Robot action/state definitions
  └── read_hdf.py      # HDF5 dataset reader
```

## Repository Structure

```
il/
├── model_load.py          # Factory: load any trained model by type
├── train_main.py          # Unified training entry point (CLI)
├── act/                   # ACT algorithm
│   ├── act_model.py       #   Inference wrapper
│   ├── act_network.py     #   Network architecture (DETR + VAE)
│   ├── act_trainer.py     #   Training with KL divergence loss
│   └── act_armpi_dataset.py  # Dataset: 100-step action chunks
├── diffusion/             # Diffusion Policy algorithm
│   ├── diffusion_model.py       # Inference with DDPM/DDIM schedulers
│   ├── diffusion_network.py     # Diffusion policy builder
│   ├── diffusion_trainer.py     # Training with noise scheduler
│   └── diffusion_armpi_dataset.py  # Dataset: horizon-based sequences
├── mlp/                   # MLP baseline
│   ├── mlp_model.py       #   Inference wrapper
│   ├── mlp_network.py     #   ResNet18 + MLP fusion
│   ├── mlp_trainer.py     #   Cross-entropy training
│   └── mlp_armpi_dataset.py  # Dataset: single-step
├── lpil/                  # LPIL algorithm
│   ├── lpil_model.py      #   Inference with goal latent
│   └── lpil_model_convert.py  # Convert training results
├── common/                # Shared utilities
│   ├── base_model.py      #   Abstract base for all models
│   ├── trainer.py         #   Base trainer (early stopping, checkpointing)
│   ├── armpi_const.py     #   Robot constants (9 actions, 6 states)
│   └── read_hdf.py        #   HDF5 dataset reader
└── third_party/           # External implementations (git submodules)
    ├── act/               #   ACT (DETR-based)
    ├── diffusion/         #   Diffusion Policy
    └── LPIL/              #   LPIL framework
```

## Quick Start

### Prerequisites

```bash
# Create conda environment
conda env create -f environment.yml
conda activate armpi_env

# Initialize submodules
git submodule update --init --recursive
```

### Training

```bash
# Train MLP baseline
python train_main.py --file_name mlp_experiment --task_name bring_up_test --model mlp --epochs 50

# Train ACT
python train_main.py --file_name act_experiment --task_name bring_up_test --model act --epochs 100

# Train Diffusion Policy
python train_main.py --file_name diffusion_experiment --task_name bring_up_test --model diffusion --epochs 100
```

**Training arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--file_name` | Name for saving the model | (required) |
| `--task_name` | Dataset folder name in `datasets/` | (required) |
| `--model` | Algorithm: `mlp`, `act`, `diffusion` | (required) |
| `--batch_size` | Batch size | 32 |
| `--val_split` | Validation split ratio | 0.2 |
| `--epochs` | Number of training epochs | 10 |
| `--learning_rate` | Learning rate | 0.001 |
| `--early_stop_patience` | Early stopping patience | 10 |

### Inference

```python
import torch
from model_load import model_load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_load('models/act_experiment', device)

# Input: camera image [3, 224, 224] + joint states [6]
image = torch.randn(3, 224, 224)
state = torch.randn(6)

# Output: 9-dimensional action vector
action = model.predict(image, state)  # -> torch.Size([9])
```

## Data Format

Training data is stored as HDF5 files in `datasets/<task_name>/`.

Each `.h5` file contains a `sync_data` key with a pandas DataFrame of synchronized:
- **Images**: Camera frames (RGB, stored as file paths)
- **States**: 6 joint positions (`joint1_pos` ... `joint5_pos`, `r_joint_pos`)
- **Actions**: 9-dimensional commands:

| Index | Action | Description |
|-------|--------|-------------|
| 0 | `chassis_move_forward` | Forward/backward |
| 1 | `chassis_move_right` | Left/right strafe |
| 2 | `angular_right` | Rotation |
| 3 | `arm_x` | Arm X position |
| 4 | `arm_y` | Arm Y position |
| 5 | `arm_z` | Arm Z position |
| 6 | `arm_alpha` | Roll orientation |
| 7 | `rotation` | Wrist rotation |
| 8 | `gripper_close` | Gripper open/close |

## Integration

This library is used as a git submodule by [ArmPi](https://github.com/shutouyusei/ArmPi) for real robot deployment. In ArmPi, it is mounted at `ros/myapp/ai_model_service/src/ai_modules/` and called by a ROS inference service node.

## License

This project is for research and educational purposes.
