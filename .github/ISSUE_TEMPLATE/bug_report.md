---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Code to reproduce**
```python
# Please provide a minimal code example that reproduces the issue
```

**Error message/traceback**
```
# Paste the full error message and traceback here
```

**Environment (please complete the following information):**
 - OS: [e.g. Ubuntu 20.04, macOS 12.0, Windows 10]
 - Python version: [e.g. 3.9.7]
 - PNO Physics Bench version: [e.g. 0.1.0]
 - PyTorch version: [e.g. 2.0.1]
 - CUDA version (if applicable): [e.g. 11.8]
 - GPU model (if applicable): [e.g. RTX 3080]

**Additional context**
Add any other context about the problem here.

**Dataset information (if applicable)**
- PDE type: [e.g. Navier-Stokes]
- Resolution: [e.g. 64x64]
- Dataset size: [e.g. 1000 samples]
- Data source: [generated/loaded from file]

**Model configuration (if applicable)**
```python
# Please provide the model configuration that causes the issue
model = ProbabilisticNeuralOperator(
    input_dim=...,
    hidden_dim=...,
    # ... other parameters
)
```