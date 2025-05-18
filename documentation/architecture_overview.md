# Architecture Overview

Simplexity is organised into modular packages. Each package focuses on a particular aspect of sequence modelling and can be used independently. The following summary highlights the main responsibilities of each module so that both developers and automated agents know where to look for functionality.

## `generative_processes`

Defines probabilistic processes used to generate training and evaluation data. Key classes include:

- **`GenerativeProcess`** – an abstract base class defining the common interface.
- **`HiddenMarkovModel`** and **`GeneralizedHiddenMarkovModel`** – concrete implementations driven by transition matrices.
- **`builder.py`** – factory helpers for constructing processes from predefined transition matrix functions in `transition_matrices.py`.

## `predictive_models`

Predictive models take sequences of observations and predict the next token. The provided implementation is:

- **`GRURNN`** – a GRU based recurrent neural network defined in `gru_rnn.py`.

The `predictive_model.py` module defines a lightweight `Protocol` specifying the call signature for models.

## `training`

Contains high level training loops built with [equinox](https://github.com/patrick-kidger/equinox) and [optax](https://github.com/deepmind/optax). Training is driven by Hydra configuration objects located in `simplexity/configs/training`. The `train_equinox_model.py` module exposes a `train()` function which performs batched data generation, loss computation and parameter updates.

## `evaluation`

Mirrors the training utilities but focuses on computing metrics such as accuracy and loss without updating model parameters. See `evaluate_equinox_model.py` for the main entry point.

## `persistence`

Facilities for saving and loading trained model weights. `LocalPersister` writes to the local file system while `S3Persister` can upload checkpoints to an S3 bucket. These are used inside the training loops.

## `logging`

A very small abstraction over experiment logging backends. Currently provided loggers include printing to stdout, writing to a file and logging to MLflow.

## `data_structures`

Simple PyTree aware data structures (`Stack`, `Queue`, `Heap`) implemented using Equinox modules. Useful when writing JAX code that manipulates collections inside `jit`ted functions.

## Utilities

The `utils` package contains helpers for JAX and Hydra integration.

## Tests

The `tests/` directory provides unit tests for most submodules. Running `pytest` is the easiest way to confirm that modifications have not broken existing functionality.

Use this overview as a starting point when navigating the code base. For extension guidelines see [Extending Simplexity](./extending.md).
