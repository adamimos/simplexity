# Using the Library

This guide walks through a minimal workflow for training and evaluating a model using Simplexity.

## Installation

1. Install [UV](https://docs.astral.sh/uv/getting-started/installation/):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install Simplexity dependencies:

   ```bash
   uv sync
   ```

   For development and testing extras run:

   ```bash
   uv sync --extra dev
   ```

   Dependencies are installed into a `.venv` directory managed by `uv`.

## Training a Model

Training is configured via [Hydra](https://hydra.cc/). The default configuration is in `simplexity/configs/train_model.yaml` and references other configuration files under `simplexity/configs/`.

Run a training session with:

```bash
uv run python simplexity/train_model.py
```

This will use a GRU based predictive model and the `mess3` generative process by default. Training logs and checkpoints are written according to the configuration. See the configuration files for adjustable parameters such as batch size, learning rate and number of steps.

## Running an Experiment

Experiments allow you to sweep over multiple parameter settings using [Optuna](https://optuna.org/). Invoke an experiment with:

```bash
uv run python simplexity/run_experiment.py --multirun
```

Hydra will spawn multiple runs according to `simplexity/configs/experiment.yaml`.

## Evaluating a Model

The `simplexity/evaluation` package contains utilities for computing loss and accuracy over a validation set. Evaluation is typically run during training but can also be invoked separately.

## Jupyter Notebooks

The `notebooks/` directory contains exploratory notebooks demonstrating how to generate data, visualise processes and inspect trained models. They are a good starting point for learning the API interactively.
