# Usage Guide - MAGI-RM

Complete guide for installing, training, evaluating, and integrating MAGI-RM into your projects.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Integration](#integration)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

## Installation

### Prerequisites

**System Requirements**:
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA 11.8+ (for GPU acceleration)
- 6GB+ VRAM (for training)
- 16GB+ system RAM

**Recommended**:
- Python 3.10+
- PyTorch 2.1+
- 2× T4/V100/A100 GPUs
- 32GB+ system RAM

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/magi-rm.git
cd magi-rm
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n magi-rm python=3.10
conda activate magi-rm
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; from transformers import AutoModel; print('✓ Installation successful')"
```

### Step 4: Download Model Weights (Optional)

```bash
# Download pre-trained OPT-1.3B base model
python scripts/download_base_model.py --model facebook/opt-1.3b
```

## Quick Start

### 30-Second Demo

```bash
# Train with default configuration
python src/training/training_dual_rm.py --config configs/rm_train_config.yaml

# Evaluate trained model
python src/eval/eval_rm.py --checkpoint runs/checkpoints/best_model.pt

# Run inference on custom input
python scripts/inference.py --checkpoint runs/checkpoints/best_model.pt \
    --input "Problem: What is 5+3? Solution: Step 1: Add 5+3=8"
```

## Training

### Basic Training

Train with default settings optimized for 2×T4 GPUs:

```bash
python src/training/training_dual_rm.py \
    --config configs/rm_train_config.yaml
```

**Expected Output**:
```
Loading base model: facebook/opt-1.3b
Initializing dual-head architecture...
Training dataset: 4679 samples
Validation dataset: 549 samples
Starting training...

Epoch 1/3:
  Step 100/585: loss=0.234, orm_acc=0.68, prm_mse=0.048 [2.1s/it]
  Step 200/585: loss=0.198, orm_acc=0.72, prm_mse=0.041 [2.0s/it]
  ...
  Validation: orm_auc=0.76, prm_corr=0.93

Training completed in 2h 48m
Best model saved to: runs/checkpoints/best_model.pt
```

### Custom Training Configuration

#### Method 1: Modify Config File

Edit `configs/rm_train_config.yaml`:

```yaml
model:
  base_model: "facebook/opt-1.3b"
  max_steps: 15
  freeze_base: true
  unfreeze_last_n_layers: 8

training:
  learning_rate: 3e-5
  num_epochs: 3
  per_device_batch_size: 4
  gradient_accumulation_steps: 8
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0

optimization:
  fp16: true
  tf32: true
  gradient_checkpointing: true

paths:
  data_dir: "./data/processed"
  output_dir: "./runs"
  
logging:
  eval_steps: 100
  save_steps: 500
  logging_steps: 10
```

#### Method 2: Command-Line Override

```bash
python src/training/training_dual_rm.py \
    --config configs/rm_train_config.yaml \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --num_epochs 5 \
    --output_dir ./runs/experiment_001 \
    --gradient_checkpointing true
```

### Multi-GPU Training

#### Using Accelerate (Recommended)

```bash
# Configure accelerate (first time only)
accelerate config

# Launch multi-GPU training
accelerate launch --multi_gpu --num_processes=2 \
    src/training/training_dual_rm.py \
    --config configs/rm_train_config.yaml
```

#### Using Torchrun

```bash
torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    src/training/training_dual_rm.py \
    --config configs/rm_train_config.yaml
```

#### Using PyTorch DDP

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    src/training/training_dual_rm.py \
    --config configs/rm_train_config.yaml
```

### Resume Training from Checkpoint

```bash
python src/training/training_dual_rm.py \
    --config configs/rm_train_config.yaml \
    --resume_from_checkpoint runs/checkpoints/checkpoint-1000
```

### Training with Custom Data

```python
# scripts/train_custom_data.py
from src.data.dataset import RewardModelDataset
from src.training.trainer import RewardModelTrainer
from transformers import AutoTokenizer

# Prepare your data
train_data = [
    {
        "problem": "What is 2+2?",
        "solution": "Step 1: Add 2+2 = 4",
        "steps": ["Add 2+2 = 4"],
        "step_rewards": [1.0],
        "final_reward": 1.0
    },
    # ... more examples
]

# Create dataset
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
dataset = RewardModelDataset(
    data=train_data,
    tokenizer=tokenizer,
    max_length=512,
    max_steps=15
)

# Train
trainer = RewardModelTrainer(config="configs/rm_train_config.yaml")
trainer.train(train_dataset=dataset)
```

## Evaluation

### Full Evaluation Suite

Run comprehensive evaluation on test set:

```bash
python src/eval/eval_rm.py \
    --checkpoint runs/checkpoints/best_model.pt \
    --data_path data/processed/test.json \
    --output_path results/evaluation_report.json \
    --batch_size 16
```

**Output** (`results/evaluation_report.json`):
```json
{
  "orm_metrics": {
    "auc": 0.778,
    "accuracy": 0.732,
    "precision": 0.745,
    "recall": 0.721,
    "f1": 0.733,
    "brier_score": 0.204
  },
  "prm_metrics": {
    "step_mse": 0.042,
    "step_mae": 0.035,
    "mean_correlation": 0.941,
    "consistency_score": 0.887
  },
  "summary": {
    "total_samples": 573,
    "avg_inference_time_ms": 12.3
  }
}
```

### Quick Evaluation (Subset)

For faster iteration during development:

```bash
python src/eval/eval_rm.py \
    --checkpoint runs/checkpoints/best_model.pt \
    --data_path data/processed/test.json \
    --max_samples 100 \
    --metrics orm_auc prm_mse prm_corr
```

### Specific Metric Evaluation

```bash
# ORM metrics only
python src/eval/eval_rm.py \
    --checkpoint runs/checkpoints/best_model.pt \
    --data_path data/processed/test.json \
    --eval_mode orm

# PRM metrics only
python src/eval/eval_rm.py \
    --checkpoint runs/checkpoints/best_model.pt \
    --data_path data/processed/test.json \
    --eval_mode prm
```

### Evaluation Metrics Explained

**ORM Metrics**:
- `auc`: Area under ROC curve (solution correctness classification)
- `accuracy`: Binary accuracy on final answers
- `brier_score`: Calibration quality (lower is better)
- `precision/recall/f1`: Standard classification metrics

**PRM Metrics**:
- `step_mse`: Mean squared error on step-level predictions
- `step_mae`: Mean absolute error on step-level predictions
- `mean_correlation`: Pearson correlation with ground truth
- `consistency_score`: Variance across steps (higher is better)

## Integration

### Basic Usage in Python

```python
from src.models.dual_head_rm import DualHeadRM
import torch

# Load trained model
model = DualHeadRM.from_pretrained(
    base_model_name="facebook/opt-1.3b",
    checkpoint_path="runs/checkpoints/best_model.pt",
    max_steps=15,
    device="cuda"
)
model.eval()

# Prepare input (example)
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Your tokenized input
attention_mask = torch.ones_like(input_ids)
step_mask = torch.tensor([[1, 1, 0, 0, 0]])  # 2 valid steps

# Get predictions
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        step_mask=step_mask
    )

# Extract rewards
orm_logit = outputs["orm_logits"]  # Shape: (1, 1)
prm_scores = outputs["prm_scores"]  # Shape: (1, 15)

# Convert to probabilities
orm_prob = torch.sigmoid(orm_logit).item()
valid_prm_scores = prm_scores[step_mask.bool()].tolist()

print(f"Solution correctness probability: {orm_prob:.3f}")
print(f"Step rewards: {valid_prm_scores}")
```

### Scoring Solutions

```python
from src.models.dual_head_rm import DualHeadRM
from transformers import AutoTokenizer
import torch

# Initialize
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = DualHeadRM.from_pretrained(
    base_model_name="facebook/opt-1.3b",
    checkpoint_path="runs/checkpoints/best_model.pt",
    device="cuda"
)
model.eval()

def score_solution(problem: str, solution: str) -> dict:
    """Score a problem-solution pair."""
    # Format input
    text = f"Problem: {problem}\n\nSolution: {solution}"
    
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Move to device
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Extract results
    correctness_logit = outputs["orm_logits"].item()
    correctness_prob = torch.sigmoid(outputs["orm_logits"]).item()
    step_rewards = outputs["prm_scores"].cpu().numpy()[0]
    
    return {
        "correctness_probability": correctness_prob,
        "correctness_logit": correctness_logit,
        "step_rewards": step_rewards.tolist(),
        "mean_step_reward": float(step_rewards.mean())
    }

# Example usage
problem = "What is (5+3)×2?"
solutions = [
    "Step 1: Add 5+3 = 8\nStep 2: Multiply 8×2 = 16",
    "Step 1: Add 5+3 = 9\nStep 2: Multiply 9×2 = 18"
]

for i, solution in enumerate(solutions, 1):
    result = score_solution(problem, solution)
    print(f"\n=== Solution {i} ===")
    print(f"Correctness: {result['correctness_probability']:.3f}")
    print(f"Mean step reward: {result['mean_step_reward']:.3f}")
```

### Batch Processing

```python
from torch.utils.data import DataLoader
from src.models.dual_head_rm import DualHeadRM

# Load model
model = DualHeadRM.from_pretrained(
    base_model_name="facebook/opt-1.3b",
    checkpoint_path="runs/checkpoints/best_model.pt",
    device="cuda"
)
model.eval()

# Create dataloader
dataloader = DataLoader(
    your_dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    collate_fn=your_collate_fn
)

# Batch inference
all_orm_scores = []
all_prm_scores = []

with torch.no_grad():
    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            step_mask=batch.get("step_mask")
        )
        
        all_orm_scores.extend(
            torch.sigmoid(outputs["orm_logits"]).cpu().tolist()
        )
        all_prm_scores.extend(outputs["prm_scores"].cpu().tolist())

print(f"Processed {len(all_orm_scores)} samples")
```

### RLHF Integration

```python
from src.models.dual_head_rm import DualHeadRM
import torch

class RLHFTrainer:
    """Example RLHF trainer using MAGI-RM for rewards."""
    
    def __init__(self, policy_model, reward_model_path):
        self.policy = policy_model
        self.reward_model = DualHeadRM.from_pretrained(
            base_model_name="facebook/opt-1.3b",
            checkpoint_path=reward_model_path,
            device="cuda"
        )
        self.reward_model.eval()
    
    def compute_rewards(self, prompts, responses):
        """Compute rewards for generated responses."""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            # Combine prompt and response
            full_text = prompt + response
            
            # Tokenize
            encoding = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.reward_model.device)
            
            # Get reward
            with torch.no_grad():
                outputs = self.reward_model(**encoding)
                # Use ORM logit as scalar reward
                reward = outputs["orm_logits"].item()
                rewards.append(reward)
        
        return torch.tensor(rewards)
    
    def train_step(self, batch):
        """Single PPO training step with reward model."""
        # Generate responses
        responses = self.policy.generate(batch["prompts"])
        
        # Compute rewards
        rewards = self.compute_rewards(batch["prompts"], responses)
        
        # PPO update (simplified)
        policy_loss = self.compute_policy_loss(responses, rewards)
        policy_loss.backward()
        self.optimizer.step()
        
        return {"loss": policy_loss.item(), "mean_reward": rewards.mean().item()}
```

### Solution Ranking

```python
def rank_solutions(problem: str, solutions: list) -> list:
    """Rank multiple solutions by predicted correctness."""
    scores = []
    
    for solution in solutions:
        result = score_solution(problem, solution)
        scores.append({
            "solution": solution,
            "score": result["correctness_probability"],
            "step_rewards": result["step_rewards"]
        })
    
    # Sort by correctness probability
    ranked = sorted(scores, key=lambda x: x["score"], reverse=True)
    return ranked

# Example
problem = "If John has 5 apples and buys 3 more, how many does he have?"
solutions = [
    "Step 1: Add 5+3 = 8",
    "Step 1: 5-3 = 2",
    "Step 1: Multiply 5×3 = 15"
]

ranked = rank_solutions(problem, solutions)
for i, item in enumerate(ranked, 1):
    print(f"{i}. Score: {item['score']:.3f}")
    print(f"   Solution: {item['solution'][:50]}...")
```

## Advanced Usage

### Custom Pooling Strategy

```python
from src.models.dual_head_rm import DualHeadRM

# Load with custom pooling
model = DualHeadRM.from_pretrained(
    base_model_name="facebook/opt-1.3b",
    checkpoint_path="runs/checkpoints/best_model.pt",
    pooling_strategy="mean",  # Options: "eos", "last", "mean", "max"
    device="cuda"
)
```

### Export for Production

#### Export to ONNX

```python
import torch
from src.models.dual_head_rm import DualHeadRM

# Load model
model = DualHeadRM.from_pretrained(
    base_model_name="facebook/opt-1.3b",
    checkpoint_path="runs/checkpoints/best_model.pt",
    device="cpu"
)
model.eval()

# Dummy input
dummy_input = {
    "input_ids": torch.randint(0, 50000, (1, 128)),
    "attention_mask": torch.ones(1, 128, dtype=torch.long)
}

# Export
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "magi_rm.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["orm_logits", "prm_scores"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=14
)
```

#### Quantization for Faster Inference

```python
import torch
from src.models.dual_head_rm import DualHeadRM

# Load model
model = DualHeadRM.from_pretrained(
    base_model_name="facebook/opt-1.3b",
    checkpoint_path="runs/checkpoints/best_model.pt",
    device="cpu"
)

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "magi_rm_quantized.pt")
```

### Fine-tuning on Custom Domain

```python
from src.training.trainer import RewardModelTrainer
from src.data.dataset import RewardModelDataset

# Load pre-trained model
trainer = RewardModelTrainer(
    config="configs/rm_train_config.yaml",
    pretrained_checkpoint="runs/checkpoints/best_model.pt"
)

# Prepare domain-specific data
custom_data = load_your_custom_data()  # Your data loading function
custom_dataset = RewardModelDataset(
    data=custom_data,
    tokenizer=trainer.tokenizer,
    max_length=512,
    max_steps=15
)

# Fine-tune
trainer.train(
    train_dataset=custom_dataset,
    learning_rate=1e-5,  # Lower LR for fine-tuning
    num_epochs=2
)
```

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

```bash
# 1. Reduce batch size
python src/training/training_dual_rm.py \
    --config configs/rm_train_config.yaml \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 16

# 2. Enable gradient checkpointing (if not already)
# In config: gradient_checkpointing: true

# 3. Use smaller model
# In config: base_model: "facebook/opt-350m"

# 4. Reduce sequence length
# In config or code: max_length: 256
```

### Slow Training

**Symptom**: Training much slower than expected

**Solutions**:

```bash
# 1. Enable TF32
export NVIDIA_TF32_OVERRIDE=1

# 2. Use torch.compile (PyTorch 2.0+)
# Add to training script before training loop:
# model = torch.compile(model, mode="reduce-overhead")

# 3. Increase number of dataloader workers
# In config: num_workers: 8

# 4. Enable persistent workers
# In dataloader: persistent_workers=True
```

### NaN Loss

**Symptom**: Loss becomes NaN during training

**Solutions**:

```yaml
# 1. Reduce learning rate
training:
  learning_rate: 1e-5  # Instead of 3e-5

# 2. Enable gradient clipping (should be on by default)
training:
  max_grad_norm: 0.5  # More aggressive

# 3. Use FP32 instead of FP16
optimization:
  fp16: false

# 4. Check data for anomalies
# Run: python scripts/validate_data.py
```

### Poor Performance

**Symptom**: Model performs worse than expected

**Checklist**:

1. **Data quality**: Run data validation
   ```bash
   python scripts/validate_data.py --data_path data/processed/train.json
   ```

2. **Training stability**: Check loss curves
   ```bash
   tensorboard --logdir runs/logs
   ```

3. **Overfitting**: Check train vs validation gap
   - If val loss >> train loss: Add regularization, reduce model size
   - If both high: Increase model capacity, train longer

4. **Learning rate**: Try different values
   ```bash
   # Sweep learning rates
   for lr in 1e-5 5e-5 1e-4; do
       python src/training/training_dual_rm.py \
           --learning_rate $lr \
           --output_dir runs/lr_$lr
   done
   ```

### Installation Issues

**Symptom**: Import errors or dependency conflicts

**Solutions**:

```bash
# 1. Clean install
pip uninstall -y torch transformers accelerate
pip install torch==2.1.0 transformers==4.35.0 accelerate==0.24.0

# 2. Check CUDA version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"

# 3. Reinstall with specific CUDA version
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Performance Optimization

### Inference Optimization

```python
import torch
from src.models.dual_head_rm import DualHeadRM

# Load model
model = DualHeadRM.from_pretrained(
    base_model_name="facebook/opt-1.3b",
    checkpoint_path="runs/checkpoints/best_model.pt",
    device="cuda"
)
model.eval()

# 1. Use torch.compile (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")

# 2. Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

# 3. Use inference mode (better than no_grad)
with torch.inference_mode():
    outputs = model(input_ids, attention_mask)

# 4. Batch requests when possible
# Process multiple inputs together instead of one at a time
```

### Memory Optimization

```python
# 1. Use gradient checkpointing in training
model.gradient_checkpointing_enable()

# 2. Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()

# 3. Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(input_ids, attention_mask)
    loss = compute_loss(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Training Optimization

```python
# Use DistributedDataParallel instead of DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=False  # Set to False for better performance
)

# Use gradient_as_bucket_view for faster gradient synchronization
model = DDP(
    model,
    device_ids=[local_rank],
    gradient_as_bucket_view=True
)
```

## Additional Resources

### Example Scripts

- `examples/basic_training.py`: Minimal training example
- `examples/custom_data.py`: Using custom datasets
- `examples/rlhf_integration.py`: RLHF pipeline example
- `examples/batch_inference.py`: Efficient batch processing

### Documentation

- [Model Architecture](docs/architecture.md)
- [Training Pipeline](docs/training.md)
- [Data Format](docs/data_format.md)
- [API Reference](docs/api.md)

### Support

- **Issues**: [GitHub Issues](https://github.com/Coder-12/magi-rm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Coder-12/magi-rm/discussions)
- **Email**: akleshmishra7@gmail.com

---

**Next Steps**:
- Review [MODEL_CARD.md](MODEL_CARD.md) for model details
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Explore [examples/](examples/) for more use cases
