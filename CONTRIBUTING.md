# Contributing to MAGI-RM

Thank you for your interest in contributing to MAGI-RM! This document provides comprehensive guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Contribution Areas](#contribution-areas)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming, inclusive, and harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behaviors include**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members
- Giving and receiving feedback professionally

**Unacceptable behaviors include**:
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate in a professional setting
- Sustained disruption of discussions

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at conduct@yourproject.com. All complaints will be reviewed and investigated promptly and fairly.

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check the [existing issues](https://github.com/yourusername/magi-rm/issues) to avoid duplicates
2. Verify you're using the latest version
3. Collect relevant information about your environment

**Bug Report Template**:

```markdown
**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.10.12]
- PyTorch Version: [e.g., 2.1.0]
- CUDA Version: [e.g., 11.8]
- GPU: [e.g., NVIDIA T4, 16GB]

**Description:**
A clear and concise description of the bug.

**To Reproduce:**
Steps to reproduce the behavior:
1. Run command: `python train_rm.py --config config.yaml`
2. Wait for epoch 2
3. Observe error

**Expected Behavior:**
What you expected to happen.

**Actual Behavior:**
What actually happened.

**Error Message:**
```
[Paste full error message and stack trace]
```

**Additional Context:**
Any other relevant information, screenshots, or logs.

**Possible Solution (Optional):**
If you have suggestions on how to fix the issue.
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

```markdown
**Enhancement Title:**
Brief, descriptive title

**Problem Statement:**
What problem does this enhancement solve?

**Proposed Solution:**
Describe your proposed solution in detail.

**Use Case:**
Provide concrete examples of how this would be used.

**Alternatives Considered:**
What alternative solutions did you consider?

**Additional Context:**
Screenshots, mockups, or examples from other projects.

**Implementation Complexity:**
- [ ] Low (< 1 day)
- [ ] Medium (1-3 days)
- [ ] High (> 3 days)
```

### Asking Questions

For questions:
- Check existing [documentation](docs/)
- Search [GitHub Discussions](https://github.com/yourusername/magi-rm/discussions)
- Open a new discussion if your question hasn't been answered

## Development Setup

### Fork and Clone

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/magi-rm.git
cd magi-rm

# 3. Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/magi-rm.git

# 4. Verify remotes
git remote -v
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

Create `requirements-dev.txt`:

```txt
# Code formatting
black==23.12.1
isort==5.13.2

# Linting
flake8==7.0.0
flake8-docstrings==1.7.0
pylint==3.0.3

# Type checking
mypy==1.8.0

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-xdist==3.5.0
pytest-timeout==2.2.0

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==2.0.0

# Git hooks
pre-commit==3.6.0

# Utilities
ipython==8.20.0
jupyter==1.0.0
```

### Pre-commit Configuration

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203]
```

### Create a Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or for bug fixes
git checkout -b fix/bug-description
```

**Branch Naming Convention**:
- `feature/`: New features
- `fix/`: Bug fixes
- `docs/`: Documentation changes
- `refactor/`: Code refactoring
- `test/`: Adding tests
- `perf/`: Performance improvements

## Contribution Areas

### 1. Model Architecture Improvements

**Areas for contribution**:
- Alternative pooling strategies (attention-weighted, learnable pooling)
- Different head architectures (transformer layers, cross-attention)
- Multi-head attention for step importance
- Hierarchical reward modeling

**Example contribution**:
```python
# In src/models/dual_head_rm.py

class ImprovedPRMHead(nn.Module):
    """Enhanced PRM head with attention mechanism."""
    
    def __init__(self, hidden_size, max_steps):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.fc = nn.Linear(hidden_size, max_steps)
    
    def forward(self, hidden_states):
        # Your improved implementation
        pass
```

### 2. Training Optimizations

**Areas for contribution**:
- Better learning rate schedules
- Advanced data augmentation techniques
- Curriculum learning strategies
- Mixed batch sampling (hard/easy examples)
- Adaptive gradient clipping

**Example contribution**:
```python
# In src/training/trainer.py

class AdaptiveScheduler:
    """Adaptive learning rate scheduler based on validation metrics."""
    
    def __init__(self, optimizer, patience=3):
        self.optimizer = optimizer
        self.patience = patience
        self.best_metric = float('inf')
        self.wait = 0
    
    def step(self, metric):
        if metric < self.best_metric:
            self.best_metric = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr()
```

### 3. Evaluation and Metrics

**Areas for contribution**:
- New evaluation metrics (ECE, MCE for calibration)
- Visualization tools for reward distributions
- Comparative analysis utilities
- Ablation study frameworks

**Example contribution**:
```python
# In src/eval/metrics.py

def expected_calibration_error(predictions, labels, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Implementation
        pass
    
    return ece
```

### 4. Data Processing

**Areas for contribution**:
- New dataset loaders (MATH, GSM8K-Hard, etc.)
- Data augmentation techniques
- Quality filtering heuristics
- Preprocessing optimizations

**Example contribution**:
```python
# In src/data/dataset.py

class MATHDataset(RewardModelDataset):
    """Dataset loader for MATH benchmark."""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        # Load MATH format
        data = self._load_math_format(data_path)
        super().__init__(data, tokenizer, max_length)
    
    def _load_math_format(self, path):
        # Implementation
        pass
```

### 5. Documentation

**Areas for contribution**:
- Tutorial notebooks
- API documentation improvements
- Architecture diagrams
- Use case examples
- Translation to other languages

### 6. Testing

**Areas for contribution**:
- Unit tests for core functionality
- Integration tests for training pipeline
- Performance regression tests
- Edge case testing

## Pull Request Process

### Before Submitting

**1. Update from upstream**:
```bash
git fetch upstream
git rebase upstream/main
```

**2. Run tests**:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_model.py::TestDualHeadRM::test_forward_pass
```

**3. Format code**:
```bash
# Auto-format with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/

# Type check (optional but recommended)
mypy src/
```

**4. Run pre-commit hooks**:
```bash
pre-commit run --all-files
```

**5. Update documentation**:
- Add/update docstrings for new functions
- Update README.md if needed
- Update CHANGELOG.md

### Creating the Pull Request

**1. Push to your fork**:
```bash
git push origin feature/your-feature-name
```

**2. Create PR on GitHub** with this template:

```markdown
## Description
Brief description of what this PR does.

## Motivation and Context
Why is this change needed? What problem does it solve?

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Changes Made
- Change 1: Description
- Change 2: Description
- Change 3: Description

## Testing
Describe the tests you ran to verify your changes.

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] New tests added for new functionality

## Test Configuration
- Python version: 3.10
- PyTorch version: 2.1.0
- GPU: T4

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Related Issues
Closes #123
Related to #456

## Additional Notes
Any additional information that reviewers should know.
```

### Review Process

1. **Automated Checks**: CI/CD will run tests, linting, and type checking
2. **Code Review**: Maintainers will review within 3-5 business days
3. **Feedback**: Address review comments by pushing new commits
4. **Approval**: Once approved, a maintainer will merge your PR
5. **Release**: Your contribution will be included in the next release

### After PR is Merged

```bash
# Update your fork
git checkout main
git pull upstream main
git push origin main

# Delete feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with modifications:

**Line Length**: 100 characters (not 79)

**Imports**:
```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

# Local imports
from src.models.base import BaseRewardModel
from src.utils.logging import get_logger
```

**Docstrings** (Google style):
```python
def compute_reward(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    step_mask: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Compute rewards using the dual-head reward model.
    
    This function performs a forward pass through the reward model and
    returns both ORM and PRM predictions.
    
    Args:
        model: The reward model (DualHeadRM instance)
        input_ids: Token IDs of shape (batch_size, seq_len)
        attention_mask: Attention mask of shape (batch_size, seq_len)
        step_mask: Optional mask for valid reasoning steps of shape
            (batch_size, max_steps). If None, all steps are considered valid.
    
    Returns:
        Dictionary containing:
            - 'orm_logits': Solution correctness logits (batch_size, 1)
            - 'prm_scores': Step-level rewards (batch_size, max_steps)
    
    Raises:
        ValueError: If input_ids and attention_mask have different shapes
        RuntimeError: If model forward pass fails
    
    Example:
        >>> model = DualHeadRM(base_model="opt-1.3b")
        >>> input_ids = torch.randint(0, 50000, (2, 128))
        >>> attention_mask = torch.ones(2, 128)
        >>> outputs = compute_reward(model, input_ids, attention_mask)
        >>> print(outputs['orm_logits'].shape)
        torch.Size([2, 1])
    """
    # Implementation
    pass
```

**Type Hints**: Always use type hints
```python
from typing import Dict, List, Optional, Tuple, Union

def process_data(
    data: List[Dict[str, any]],
    max_length: int = 512,
    return_tensors: bool = True
) -> Union[Dict[str, torch.Tensor], List[Dict]]:
    """Process data with type hints."""
    pass
```

**Naming Conventions**:
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- `_leading_underscore` for private methods

**Code Organization**:
```python
"""Module docstring.

This module implements the dual-head reward model architecture.
"""

# Imports
import torch
import torch.nn as nn

# Constants
MAX_SEQUENCE_LENGTH = 512
DEFAULT_HIDDEN_SIZE = 2048

# Classes
class DualHeadRM(nn.Module):
    """Dual-head reward model class."""
    
    def __init__(self, base_model: str):
        """Initialize the model."""
        super().__init__()
        # Implementation
    
    def forward(self, input_ids, attention_mask):
        """Forward pass."""
        # Implementation

# Functions
def load_model(checkpoint_path: str) -> DualHeadRM:
    """Load model from checkpoint."""
    # Implementation
```

## Testing Guidelines

### Writing Tests

**Test Structure**:
```python
# tests/test_model.py
import pytest
import torch
from src.models.dual_head_rm import DualHeadRM

class TestDualHeadRM:
    """Test suite for DualHeadRM model."""
    
    @pytest.fixture
    def model(self):
        """Create a small model instance for testing."""
        return DualHeadRM(
            base_model_name="facebook/opt-125m",  # Small model
            max_steps=5,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensors."""
        batch_size, seq_len = 2, 64
        return {
            "input_ids": torch.randint(0, 50000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "step_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
        }
    
    def test_forward_pass_shape(self, model, sample_input):
        """Test that forward pass produces correct output shapes."""
        outputs = model(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
            step_mask=sample_input["step_mask"]
        )
        
        assert "orm_logits" in outputs
        assert "prm_scores" in outputs
        assert outputs["orm_logits"].shape == (2, 1)
        assert outputs["prm_scores"].shape == (2, 5)
    
    def test_gradient_flow(self, model, sample_input):
        """Test that gradients flow through both heads."""
        outputs = model(**sample_input)
        loss = outputs["orm_logits"].sum() + outputs["prm_scores"].sum()
        loss.backward()
        
        # Check gradients exist for all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    def test_deterministic_output(self, model, sample_input):
        """Test that model produces deterministic outputs."""
        torch.manual_seed(42)
        outputs1 = model(**sample_input)
        
        torch.manual_seed(42)
        outputs2 = model(**sample_input)
        
        torch.testing.assert_close(outputs1["orm_logits"], outputs2["orm_logits"])
        torch.testing.assert_close(outputs1["prm_scores"], outputs2["prm_scores"])
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_variable_batch_size(self, model, batch_size):
        """Test model with different batch sizes."""
        input_ids = torch.randint(0, 50000, (batch_size, 64))
        attention_mask = torch.ones(batch_size, 64)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert outputs["orm_logits"].shape[0] == batch_size
    
    def test_invalid_input(self, model):
        """Test model handles invalid input gracefully."""
        with pytest.raises(ValueError):
            model(
                input_ids=torch.randint(0, 50000, (2, 64)),
                attention_mask=torch.ones(2, 32)  # Wrong shape
            )
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestDualHeadRM::test_forward_pass_shape

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run in parallel
pytest tests/ -n auto

# Run with timeout
pytest tests/ --timeout=300
```

### Test Coverage

We aim for >80% code coverage. Check coverage:

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Coverage Requirements**:
- Core model code: >90%
- Training pipeline: >80%
- Utilities: >70%
- Scripts: >50%

## Documentation

### Docstring Standards

Every public function, class, and module must have docstrings:

```python
def train_reward_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 3e-5,
    device: str = "cuda"
) -> Dict[str, float]:
    """Train the reward model using the provided data loaders.
    
    This function implements the complete training loop with validation,
    checkpointing, and early stopping. It uses mixed precision training
    and gradient accumulation for memory efficiency.
    
    Args:
        model: The reward model to train (DualHeadRM instance)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs (default: 3)
        learning_rate: Learning rate for AdamW optimizer (default: 3e-5)
        device: Device to train on, 'cuda' or 'cpu' (default: 'cuda')
    
    Returns:
        Dictionary containing final training metrics:
            - 'final_train_loss': Final training loss
            - 'best_val_loss': Best validation loss achieved
            - 'best_val_auc': Best validation AUC achieved
            - 'total_steps': Total number of training steps
    
    Raises:
        ValueError: If num_epochs is not positive
        RuntimeError: If CUDA out of memory during training
        FileNotFoundError: If checkpoint directory cannot be created
    
    Example:
        >>> model = DualHeadRM(base_model="opt-1.3b")
        >>> train_loader = DataLoader(train_dataset, batch_size=4)
        >>> val_loader = DataLoader(val_dataset, batch_size=8)
        >>> metrics = train_reward_model(
        ...     model, train_loader, val_loader, num_epochs=3
        ... )
        >>> print(f"Best validation AUC: {metrics['best_val_auc']:.3f}")
        Best validation AUC: 0.782
    
    Note:
        This function modifies the model in-place. To preserve the original
        model, create a deep copy before training.
    
    See Also:
        evaluate_model: For evaluating a trained model
        load_checkpoint: For loading a saved checkpoint
    """
    # Implementation
    pass
```

### Updating Documentation

When making changes:

1. **Update docstrings** for modified functions/classes
2. **Update README.md** if installation or usage changes
3. **Update USAGE.md** for new features or changed APIs
4. **Update MODEL_CARD.md** if model capabilities change
5. **Add examples** to `examples/` directory
6. **Update CHANGELOG.md** with your changes

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

## Community

### Getting Help

- **Questions**: [GitHub Discussions - Q&A](https://github.com/yourusername/magi-rm/discussions/categories/q-a)
- **Bugs**: [GitHub Issues](https://github.com/yourusername/magi-rm/issues)
- **Security**: Email security@yourproject.com
- **General**: [GitHub Discussions - General](https://github.com/yourusername/magi-rm/discussions/categories/general)

### Communication Channels

- **GitHub Discussions**: Primary discussion forum
- **Issue Tracker**: Bug reports and feature requests
- **Pull Requests**: Code contributions and reviews

### Recognition

Contributors will be:
- Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Mentioned in release notes for their contributions
- Acknowledged in academic papers if applicable
- Given credit in documentation

## License

By contributing to MAGI-RM, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to MAGI-RM!**

Your contributions help advance research in reward modeling and AI alignment. Whether you're fixing a typo or implementing a new feature, every contribution is valuable and appreciated.

For questions about contributing, feel free to open a discussion or reach out to the maintainers.