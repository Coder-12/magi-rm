# Release Notes — MagiCore-RM

## Version 1.0.0 — Public Launch (Stable)

This release introduces:

### 🎯 Core Features
- Dual-Head Reward Model (ORM + PRM)
- PRM step-wise scoring with sigmoid-stabilized logits
- Outcome Reward Model with correct-or-not semantics
- Data integrity checks across 20+ QC metrics

### 🔧 Engineering Improvements
- FP16-stable forward pass (clamp, dtype normalization)
- Selective unfreezing (last 8 layers) for ORM repair
- Reward model training under 6GB GPU utilization
- Balanced ORM sampling & deterministic PRM target loader

### 🧪 Evaluation & Benchmarks
- PRM Step MSE: ~0.039
- PRM Corr: ~0.94
- ORM Accuracy: 0.73
- Brier Score: 0.19–0.21

### 🪢 Tooling
- Dataset Audits
- Lambda auto-calibration reports
- Step-length analysis tools

This version is ready for:
- Academic replication
- Industry-grade demos
- Recruiter / hiring manager evaluation