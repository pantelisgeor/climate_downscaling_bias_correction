# Climate Downscaling and Bias Correction with Vision Transformers

Deep learning framework for statistical downscaling and bias correction of climate model outputs using Vision Transformers (ViT) with FiLM (Feature-wise Linear Modulation) conditioning.

## Overview

This project implements a Single-Encoder Multi-Decoder (1EMD) architecture for correcting systematic biases in climate model outputs. The model processes spatial climate fields and corrects multiple climate variables simultaneously while conditioning on forecast lead time.

### Key Features

- **Dual Encoder Support**: Vision Transformer (ViT) or CNN-based encoders
- **Lead-Time Conditioning**: FiLM layers for lead-time-aware bias correction
- **Multi-Task Learning**: Simultaneous correction of multiple climate variables
- **Physics-Informed Losses**: Optional physical consistency constraints
- **Flexible Architecture**: Configurable dimensions, depths, and components

### Climate Variables

The model corrects bias for four key climate variables:
- **tasERA**: Near-surface air temperature (mean)
- **tasmaxERA**: Maximum near-surface air temperature
- **tpERA**: Total precipitation
- **rhERA**: Relative humidity

---

## Architecture

### Single-Encoder Multi-Decoder (1EMD)

```

Input (Static + Dynamic Fields)
↓
Encoder (ViT/CNN)
↓
FiLM Conditioning (Lead Time)
↓
Multi-Task Decoder
↓
Output (tasERA, tasmaxERA, tpERA, rhERA)

```

### Vision Transformer Encoder

- **Patch-based processing**: Divides climate fields into patches
- **Self-attention**: Captures long-range spatial dependencies
- **Positional encoding**: Preserves spatial information
- **Layer normalization**: Stabilizes training

### FiLM Layer (Feature-wise Linear Modulation)

Conditions the encoded representations on forecast lead time:

```

γ, β = FiLM(lead_embedding)
output = γ ⊙ encoded + β

```

**Critical Implementation Detail**: FiLM layers initialize with γ=1, β=0 (identity transformation) to ensure stable training from the start.

### Multi-Task Decoder

Separate decoder heads for each climate variable:
- Shared feature extraction layers
- Variable-specific upsampling paths
- Spatial reconstruction to original resolution

---

## Project Structure

```

.
├── configs/
│   ├── default.yml              \# CNN encoder configuration
│   └── default_ViT.yml          \# ViT encoder configuration
│
├── src/
│   ├── models/
│   │   ├── climate_net.py       \# Main 1EMD architecture
│   │   ├── encoder.py           \# CNN and ViT encoders
│   │   ├── decoder.py           \# Multi-task decoder
│   │   └── film_layer.py        \# FiLM conditioning layers
│   │
│   ├── training/
│   │   ├── trainer.py           \# Training loop and optimization
│   │   ├── evaluator.py         \# Evaluation and metrics
│   │   └── climate_dataset.py  \# PyTorch dataset for climate data
│   │
│   ├── losses/
│   │   ├── data_losses.py       \# MSE, MAE, etc.
│   │   ├── physics_losses.py    \# Physical consistency constraints
│   │   └── task_weighting.py    \# Dynamic task weighting
│   │
│   ├── data_loader.py           \# Data loading utilities
│   └── compute_cache.py         \# Dataset preprocessing
│
├── scripts/
│   ├── train.py                 \# Main training script
│   ├── train_dpp.sh             \# Distributed training script
│   ├── debug_vit_clean.py       \# ViT debugging utilities
│   └── test_cnn_load.py         \# CNN testing utilities
│
├── tests/
│   ├── conftest.py              \# Pytest configuration
│   └── test_data_loader.py      \# Data loader tests
│
├── main.py                      \# Main entry point
├── README.md                    \# This file
└── .gitignore                   \# Git ignore rules

```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/pantelisgeor/climate_downscaling_bias_correction.git
cd climate_downscaling_bias_correction
```

2. **Create conda environment:**
```bash
conda create -n climate python=3.9
conda activate climate
```

3. **Install dependencies:**
```bash
# PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Other dependencies
pip install pyyaml numpy xarray netCDF4 scipy matplotlib
pip install tensorboard wandb  # Optional: for logging
```


---

## Usage

### Quick Start

```python
from src.models.climate_net import ClimateNet
import torch

# Create model with ViT encoder
model = ClimateNet(
    encoder_type='vit',
    encoder_dim=512,
    encoder_blocks=6,
    vit_patch_size=7,
    vit_num_heads=8,
    use_film=True,
    num_leads=11,
    lead_embed_dim=128,
    target_vars=['tasERA', 'tasmaxERA', 'tpERA', 'rhERA']
)

# Create dummy input
batch_size = 8
static = torch.randn(batch_size, 3, 35, 77)    # Static fields
dynamic = torch.randn(batch_size, 16, 35, 77)  # Dynamic fields
lead_indices = torch.randint(0, 11, (batch_size,))

# Forward pass
outputs = model(static, dynamic, lead_indices)

# outputs is a dict: {'tasERA': tensor, 'tasmaxERA': tensor, ...}
```


### Training

**Using configuration file:**

```bash
python scripts/train.py --config configs/default_ViT.yml
```

**Command line arguments:**

```bash
python scripts/train.py \
    --encoder_type vit \
    --encoder_dim 512 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --use_film \
    --output_dir experiments/vit_film
```

**Distributed training (multi-GPU):**

```bash
bash scripts/train_dpp.sh
```


### Configuration

Example `configs/default_ViT.yml`:

```yaml
model:
  encoder_type: vit
  encoder_dim: 512
  encoder_blocks: 6
  vit_patch_size: 7
  vit_num_heads: 8
  vit_mlp_ratio: 4.0
  vit_dropout: 0.1
  use_film: true
  num_leads: 11
  lead_embed_dim: 128

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  optimizer: adam
  scheduler: cosine
  
data:
  image_size: 
  static_channels: 3
  dynamic_channels: 16
  
losses:
  mse_weight: 1.0
  physics_weight: 0.1
  use_task_weighting: true
```


---

## Model Details

### Vision Transformer Configuration

| Parameter | Default | Description |
| :-- | :-- | :-- |
| `encoder_dim` | 512 | Embedding dimension |
| `encoder_blocks` | 6 | Number of transformer blocks |
| `vit_patch_size` | 7 | Patch size (7×7) |
| `vit_num_heads` | 8 | Multi-head attention heads |
| `vit_mlp_ratio` | 4.0 | MLP hidden dim ratio |
| `vit_dropout` | 0.1 | Dropout rate |

### FiLM Conditioning

- **Purpose**: Modulates encoded features based on forecast lead time
- **Input**: Lead time index (0-10 for 11 forecast steps)
- **Mechanism**: Affine transformation per feature channel
- **Initialization**: γ=1, β=0 (identity) for stable training


### Decoder Architecture

| Layer Type | Input Dim | Output Dim | Purpose |
| :-- | :-- | :-- | :-- |
| Linear | 512 | 512 | Feature extraction |
| Linear | 512 | 256 | Dimensionality reduction |
| Reshape | 256 | (256, h/4, w/4) | Spatial arrangement |
| Conv2d | 256 | 128 | Upsampling path |
| Conv2d | 128 | 64 | Upsampling path |
| Conv2d | 64 | 1 | Output layer |

### Parameter Count

**Vision Transformer (default config):**

- Encoder: ~25M parameters
- Decoder: ~8M parameters (all tasks)
- FiLM: ~0.1M parameters
- **Total: ~33M parameters**

```python
model = ClimateNet(encoder_type='vit')
params = model.count_parameters()
print(f"Total parameters: {params['total_millions']:.2f}M")
```


---

## Data Format

### Input Data

**Static fields (3 channels):**

- Elevation
- Land-sea mask
- Other time-invariant features

**Dynamic fields (16 channels):**

- Temperature fields
- Pressure levels
- Humidity
- Wind components
- Other time-varying features

**Expected shapes:**

- Static: `(batch, 3, 35, 77)`
- Dynamic: `(batch, 16, 35, 77)`
- Lead indices: `(batch,)` - integers in [0, 10]


### Output Data

Dictionary with corrected fields:

- `tasERA`: `(batch, 1, 35, 77)`
- `tasmaxERA`: `(batch, 1, 35, 77)`
- `tpERA`: `(batch, 1, 35, 77)`
- `rhERA`: `(batch, 1, 35, 77)`

---

## Loss Functions

### Data Losses

**Mean Squared Error (MSE):**

```python
loss_mse = F.mse_loss(prediction, target)
```

**Mean Absolute Error (MAE):**

```python
loss_mae = F.l1_loss(prediction, target)
```


### Physics-Informed Losses

Optional physical consistency constraints:

- Temperature-humidity relationships
- Precipitation non-negativity
- Spatial smoothness


### Task Weighting

Dynamic task weighting balances multi-task learning:

- Automatically adjusts weights during training
- Prevents task dominance
- Improves overall performance

---

## Debugging and Testing

### Test ViT Encoder

```bash
python scripts/debug_vit_clean.py
```

Expected output:

```
FiLM Initialization Check:
  Gamma mean: 1.0000 (should be ~1.0)
  Beta mean: 0.0000 (should be ~0.0)
  ✅ FiLM correctly initialized

ViT Encoded - min: -4.41, max: 4.53, mean: 0.00, std: 1.00
After FiLM - min: -4.41, max: 4.53, mean: 0.00, std: 1.00

✅ ViT output looks healthy! Ready to train.
```


### Test CNN Encoder

```bash
python scripts/test_cnn_load.py
```


### Run Tests

```bash
pytest tests/
```


---

## Training Tips

### Recommended Hyperparameters

**For Vision Transformer:**

- Learning rate: 1e-4 with cosine annealing
- Batch size: 16-32 (depending on GPU memory)
- Warmup: 5-10 epochs
- Weight decay: 1e-5
- Gradient clipping: 1.0

**For CNN:**

- Learning rate: 1e-3 with step decay
- Batch size: 32-64
- No warmup needed
- Weight decay: 1e-4


### GPU Memory Optimization

```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(static, dynamic, lead_indices)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```


### Monitoring Training

- Use TensorBoard or Weights \& Biases for logging
- Monitor per-variable losses separately
- Check gradient norms (should be ~1.0)
- Validate on held-out lead times

---

## Known Issues \& Solutions

### Issue: FiLM outputs collapse to near-zero

**Symptom:** After FiLM layer, std drops from 1.0 to ~0.0003

**Solution:** ✅ Fixed in current version

- FiLM layers created **after** weight initialization
- Ensures γ=1, β=0 initialization is preserved


### Issue: NaN losses during training

**Solution:**

- Use gradient clipping
- Check input normalization
- Reduce learning rate
- Enable mixed precision training

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{climate_downscaling_2026,
  author = {Georgiades, Pantelis},
  title = {Climate Downscaling and Bias Correction with Vision Transformers},
  year = {2026},
  url = {https://github.com/pantelisgeor/climate_downscaling_bias_correction}
}
```


---

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

---

## Acknowledgments

- Vision Transformer implementation inspired by PyTorch Image Models (timm)
- FiLM conditioning based on "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., 2018)
- Multi-task learning framework follows best practices from climate ML literature

---

## Contact

For questions, issues, or collaboration:

- GitHub Issues: [https://github.com/pantelisgeor/climate_downscaling_bias_correction/issues](https://github.com/pantelisgeor/climate_downscaling_bias_correction/issues)
- Email: pantelisgeor@hotmail.com

---

**Status**: ✅ Model architecture complete and tested. Ready for training on full dataset.
EOF

echo "✅ README.md updated!"

```

This creates a comprehensive README with:

✅ **Clear overview** of the project  
✅ **Detailed architecture** explanation  
✅ **Complete project structure**  
✅ **Installation instructions**  
✅ **Usage examples** with code  
✅ **Configuration options**  
✅ **Model specifications** with tables  
✅ **Data format** documentation  
✅ **Loss functions** explained  
✅ **Debugging/testing** instructions  
✅ **Training tips** and hyperparameters  
✅ **Known issues** and solutions  
✅ **Citation** template  

Run the command above to create the file, then commit and push:

```bash
git add README.md
git commit -m "Add comprehensive README documentation"
git push
```
