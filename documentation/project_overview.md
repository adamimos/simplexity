# Project Overview

Simplexity is a Python library for exploring sequence prediction models from a Computational Mechanics perspective. It contains implementations of generative processes, predictive models and utilities for training, evaluating and persisting models. The code targets Python 3.12 and makes heavy use of `jax` and `equinox` for high performance computation.

This document summarises the main areas of the project so that new users and automated agents can easily locate relevant modules.

## Repository Layout

```
README.md          - Quick start instructions
notebooks/         - Example Jupyter notebooks
simplexity/        - Core library code
  configs/         - Hydra configuration files
  data_structures/ - Queue, Stack and related utilities
  evaluation/      - Model evaluation helpers
  generative_processes/ - Hidden Markov Models and related processes
  logging/         - Simple logging interfaces
  persistence/     - Save and load model weights
  predictive_models/ - Example models such as GRU RNNs
  training/        - Training loops
  utils/           - Small helper functions
tests/             - Unit tests
```

If you are completely new to the project, start with [Using the Library](./using_library.md) which explains how to install dependencies and run a simple training session.
