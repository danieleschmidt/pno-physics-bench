PNO Physics Bench Documentation
===============================

Welcome to PNO Physics Bench, a comprehensive training and benchmark suite for 
Probabilistic Neural Operators (PNO) that quantify uncertainty in PDE surrogates.

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white
   :target: https://pytorch.org/
   :alt: PyTorch

Overview
--------

**pno-physics-bench** implements the groundbreaking Probabilistic Neural Operators 
from the February 2025 arXiv paper, providing the first comprehensive framework for 
uncertainty quantification in neural PDE solvers. Unlike deterministic neural operators, 
PNOs capture both aleatoric and epistemic uncertainty, crucial for safety-critical 
applications in engineering and scientific computing.

Key Features
-----------

* **Uncertainty Quantification**: Rigorous probabilistic predictions for PDE solutions
* **Multiple Baselines**: FNO, TNO, DeepONet implementations for comparison  
* **Coverage Metrics**: Novel evaluation metrics for uncertainty calibration
* **Rollout BoE**: Bounds on error propagation for long-term predictions

Quick Start
----------

.. code-block:: bash

   # Install the package
   pip install pno-physics-bench
   
   # Or install from source
   git clone https://github.com/yourusername/pno-physics-bench.git
   cd pno-physics-bench
   pip install -e ".[dev]"

Basic Usage
----------

.. code-block:: python

   from pno_physics_bench import ProbabilisticNeuralOperator, PDEDataset
   from pno_physics_bench.training import PNOTrainer
   
   # Load dataset
   dataset = PDEDataset.load("navier_stokes_2d", resolution=64)
   train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)
   
   # Initialize PNO
   model = ProbabilisticNeuralOperator(
       input_dim=3,
       hidden_dim=256,
       num_layers=4,
       modes=20,
       uncertainty_type="full",
       posterior="variational"
   )
   
   # Train with uncertainty-aware loss
   trainer = PNOTrainer(model=model, learning_rate=1e-3)
   trainer.fit(train_loader, val_loader, epochs=100)

Table of Contents
================

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/models
   api/training
   api/datasets
   api/uncertainty
   api/metrics
   api/visualization

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics
   
   advanced/physics_informed
   advanced/active_learning
   advanced/deployment
   advanced/benchmarking

.. toctree::
   :maxdepth: 2
   :caption: Development
   
   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`