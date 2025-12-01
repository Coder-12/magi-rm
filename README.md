# MAGI-RM: Dual-Head Reward Modeling System
```
███╗   ███╗ █████╗  ██████╗ ██╗      ██████╗ ███╗   ███╗
████╗ ████║██╔══██╗██╔════╝ ██║      ██╔══██╗████╗ ████║
██╔████╔██║███████║██║  ███╗██║█████╗██████╔╝██╔████╔██║
██║╚██╔╝██║██╔══██║██║   ██║██║╚════╝██╔══██╗██║╚██╔╝██║
██║ ╚═╝ ██║██║  ██║╚██████╔╝██║      ██║  ██║██║ ╚═╝ ██║
╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝      ╚═╝  ╚═╝╚═╝     ╚═╝
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A research-grade reward modeling system implementing both Outcome Reward Modeling (ORM) and Process Reward Modeling (PRM) in a unified architecture. Designed for LLM reasoning tasks with GPU-efficient training optimizations for commodity hardware.

## Overview

MAGI-RM implements a dual-head architecture inspired by recent advances in reward modeling from Anthropic, DeepSeek R1, and the MAGICORE pipeline. The system provides:

- **ORM (Outcome Reward Model)**: Solution-level correctness scoring
- **PRM (Process Reward Model)**: Step-level reasoning reward prediction

The architecture is optimized to run efficiently on standard GPUs (T4/V100) while maintaining research-grade performance on mathematical reasoning tasks like GSM8K.

## Key Features

### Architecture
- Dual-head design with shared base encoder (OPT 1.3B)
- Step-aware PRM head supporting up to 15 reasoning steps
- Solution-level ORM classifier for final answer evaluation
- Dynamic masking for variable-length reasoning chains

### Training Optimizations
- **Memory efficiency**: Activation checkpointing reduces VRAM usage by ~40%
- **Mixed precision**: FP16 training with TF32 compute kernels
- **Selective fine-tuning**: Unfreeze only last N layers of base model
- **Gradient stability**: Logits clamping (±15) and NaN-resistant normalization
- **Optional EMA**: Exponential moving average for smoother convergence

### Evaluation Suite
- **ORM metrics**: AUC, accuracy, Brier score, logit statistics
- **PRM metrics**: Step-wise MSE/MAE, aggregated reward correlation
- **Consistency analysis**: Cross-step variance and stability metrics
- **Dataset diagnostics**: Entropy, skewness, JSD drift analysis

## Dataset

The system is trained on a curated GSM8K-based reward modeling dataset with the following characteristics:

| Split | Samples | PRM Mean | ORM Ratio |
|-------|---------|----------|-----------|
| Train | 4,679 | 0.77 | 0.73 |
| Val | 549 | 0.79 | 0.75 |
| Test | 573 | 0.78 | 0.74 |

**Quality metrics:**
- Entropy: 1.6–1.7 bits (healthy diversity)
- JSD across splits: 0.055–0.066 (excellent consistency)
- Skewness: ~-1.78 (appropriate for confidence modeling)
- Composite Quality Score: 0.676/1.0

The dataset demonstrates clean splits with no overlap, balanced supervision signals, and stable reward distributions suitable for robust RM training.

## Architecture Details

```
DualHeadRM
├── Base Encoder (OPT 1.3B, FP16)
│   ├── Gradient checkpointing enabled
│   ├── Layer-selective unfreezing
│   └── Mixed-precision compute (FP16/TF32)
├── ORM Head
│   └── 2-layer MLP → Single logit (correctness score)
└── PRM Head
    └── 2-layer MLP → max_steps logits (per-step rewards)
```

**Pooling strategy**: EOS token or last token with automatic fallback handling for robust sequence representation.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/magi-rm.git
cd magi-rm

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Accelerate 0.20+
- Additional dependencies listed in `requirements.txt`

## Quick Start

### Training

```bash
python src/training/training_dual_rm.py \
    --config configs/rm_train_config.yaml
```

**Training configuration highlights:**
- Effective batch size: 64 (4 per device × 8 accumulation steps)
- Optimized for 2×T4 or equivalent GPUs
- Mixed precision training enabled by default

### Evaluation

```bash
python src/eval/eval_rm.py \
    --checkpoint runs/checkpoints/best_model.pt
```

## Expected Performance

Based on architecture design, dataset characteristics, and optimization pipeline:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| ORM AUC | 0.72 – 0.82 | Typical for quality RMs on GSM8K |
| PRM Step-MSE | 0.035 – 0.055 | Low noise, stable variance |
| PRM Mean-Correlation | 0.92 – 0.96 | Dataset shows ~0.94 baseline |
| Reward Consistency | Stable | Balanced variance-entropy prevents collapse |

These estimates align with published results from similar architectures (MAGICORE-PRM, DeepSeek-R1) trained on comparable datasets.

## Repository Structure

```
project-root/
├── configs/
│   └── rm_train_config.yaml          # Training hyperparameters
├── data/
│   └── processed/                    # Train/val/test splits
├── src/
│   ├── models/
│   │   └── dual_head_rm.py          # Model architecture
│   └── training/
│       └── training_dual_rm.py      # Training loop
├── runs/
│   └── checkpoints/                  # Saved models
├── requirements.txt
└── README.md
```

## Roadmap

- **v1.0**: Initial public release with core architecture and training pipeline
- **v1.1**: Enhanced ORM sampling strategies and data balancing
- **v2.0**: RLHF integration with PPO and R1-style policy updates
- **v3.0**: Distilled lightweight models for efficient inference-time scaling

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. For major changes, please open an issue first to discuss proposed modifications.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{mishra2025magirm,
  author = {Mishra, Aklesh},
  title = {MAGI-RM: Dual-Head Reward Modeling for LLM Reasoning},
  year = {2025},
  url = {https://github.com/yourusername/magi-rm}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Aklesh Mishra** - Lead Research Engineer & Architect

---

**Note**: This is a research project. While the architecture and training pipeline are production-ready, the models are intended for research purposes and should be validated for specific production use cases.