# MAGI-RM — Dual-Head Reward Modeling for LLM Reasoning

```
███╗   ███╗ █████╗  ██████╗ ██╗      ██████╗ ███╗   ███╗
████╗ ████║██╔══██╗██╔════╝ ██║      ██╔══██╗████╗ ████║
██╔████╔██║███████║██║  ███╗██║█████╗██████╔╝██╔████╔██║
██║╚██╔╝██║██╔══██║██║   ██║██║╚════╝██╔══██╗██║╚██╔╝██║
██║ ╚═╝ ██║██║  ██║╚██████╔╝██║      ██║  ██║██║ ╚═╝ ██║
╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝      ╚═╝  ╚═╝╚═╝     ╚═╝
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](./README.md)

MAGI-RM is a research pipeline for reward modeling of LLM reasoning chains. A single shared encoder feeds two heads: a solution-level **Outcome Reward Model (ORM)** that predicts whether a complete solution is correct, and a step-level **Process Reward Model (PRM)** that scores each reasoning step (up to 15). The training code (`src/training/training_dual_rm.py`) supports gradient checkpointing, fp16 mixed precision, optional EMA, DDP and checkpoint resume; the data is a GSM8K-derived set of labeled reasoning chains.

> **Status: experimental.** No trained-model evaluation metrics are committed to this repository yet (see [Results](#results)).

## Quick start

```bash
git clone https://github.com/Coder-12/magi-rm.git && cd magi-rm
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # requires Python >= 3.10
pip install pytest matplotlib                         # needed only by the smoke tests

# Smoke tests (CPU, no downloads): loss forward/backward and gradient balance
python tests/test_prm_orm_loss_smoke.py && python tests/test_prm_orm_loss_smoke_extended.py
```

The smoke tests are standalone scripts that run at import time and contain no `test_*` functions, so `pytest tests -q` collects 0 tests; run the scripts directly as above. The extended script rewrites the three PNGs in `tests/`.

Train (needs a GPU and the base model path set in the config):

```bash
python src/training/training_dual_rm.py configs/rm_train_config.yaml
```

## Results

No model-evaluation metrics (accuracy, AUC, step MSE, correlation) are committed in this repository: `runs/*/metrics.jsonl` hold only a creation timestamp. What is committed is the dataset split summary:

| Split | Records | ORM positive-label ratio | Mean PRM step score |
|-------|---------|--------------------------|---------------------|
| train | 4,679 | 0.754 | 0.796 |
| val   | 549   | 0.730 | 0.782 |
| test  | 573   | 0.756 | 0.781 |

Source: [`data/processed/split_summary.json`](./data/processed/split_summary.json) (record counts match `wc -l` on `data/processed/orm_{train,val,test}.jsonl`).

---

## Key features

- Dual-head model with a shared encoder, ORM head and PRM head: [`src/models/dual_head_rm.py`](./src/models/dual_head_rm.py)
- Gradient checkpointing, fp16 mixed precision, optional EMA (`use_ema` in [`configs/rm_train_config.yaml`](./configs/rm_train_config.yaml))
- DDP support and Accelerate config: [`src/training/training_dual_rm.py`](./src/training/training_dual_rm.py), [`configs/accelerate_config.yaml`](./configs/accelerate_config.yaml)
- OOM retry with reduced batch size, and checkpoint resume (`resume_from`) in the training loop
- Dataset QC reports in [`data/qc/`](./data/qc) and split integrity reports in [`data/processed/`](./data/processed)

## Repository structure

```
magi-rm/
├── src/            # models/, training/, data/, eval/, losses/, utils/
├── configs/        # training and Accelerate configs
├── scripts/        # data collection, labeling, QC
├── data/           # collected chains, splits, processed ORM/PRM sets, QC reports
├── runs/           # run metadata (config/env)
├── tests/          # CPU smoke tests
├── MODEL_CARD.md  USAGE.md  CHANGELOG.md  RELEASE_NOTES.md
└── LICENCE         # MIT
```

---

## 🗺️ Roadmap

- **v1.0** (Current) — Public release with core architecture, training pipeline, and evaluation suite
- **v1.1** (Planned) — ORM class rebalancing, LoRA fine-tuning adapters, improved negative sampling
- **v2.0** (Future) — RLHF integration for policy optimization experiments
- **v3.0** (Future) — Distilled lightweight RM for production inference

---

## 📚 Citation

If you use MAGI-RM in your research, please cite:

```bibtex
@software{mishra2025magirm,
  author = {Mishra, Aklesh},
  title = {MAGI-RM: Dual-Head Reward Modeling for LLM Reasoning},
  year = {2025},
  url = {https://github.com/Coder-12/magi-rm},
  note = {Research-grade reward modeling with dual-head architecture}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for guidelines.

**Areas for Contribution:**
- Additional evaluation benchmarks
- Alternative base model architectures
- Dataset augmentation strategies
- Inference optimization techniques
- Documentation improvements

---

## 📄 License & Ethics

**License:** MIT License (see [`LICENCE`](./LICENCE))

**Ethical Considerations:**
- Review [`MODEL_CARD.md`](./MODEL_CARD.md) for intended use cases and limitations
- See [`SECURITY.md`](./SECURITY.md) for safety guidelines and responsible disclosure
- This model is designed for research purposes and requires careful evaluation before production deployment
- Consider potential biases inherited from base models and training data

---

## 📞 Contact & Acknowledgments

**Lead Developer:** Aklesh Mishra  
**Contact:** Via [GitHub profile](https://github.com/Coder-12)

**Acknowledgments:**  
This work builds on excellent open-source foundations:
- [Hugging Face Transformers](https://huggingface.co/transformers/) for model implementations
- [PyTorch](https://pytorch.org/) for deep learning infrastructure
- [GSM8K](https://github.com/openai/grade-school-math) for dataset inspiration
- The broader ML research community for insights on reward modeling

---

## 📖 Additional Resources

- **[MODEL_CARD.md](./MODEL_CARD.md)** — Detailed model card with architecture, training details, and risk analysis
- **[USAGE.md](./USAGE.md)** — Inference examples, API usage, and deployment patterns
- **[CONTRIBUTING.md](./CONTRIBUTING.md)** — Contribution guidelines and development setup

---

**Status:** ✨ Experimental Research Release  
**Last Updated:** December 2025  
**Version:** 1.0.0
