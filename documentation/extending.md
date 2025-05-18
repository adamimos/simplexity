# Extending Simplexity

This page provides guidance for developers and automated agents who need to add new functionality to the library. Because the codebase relies heavily on JAX and Equinox, many objects are functional and immutable by design. Follow the patterns below to keep new features consistent with the existing style.

## Adding a New Generative Process

1. Create transition matrices in `simplexity/generative_processes/transition_matrices.py`. Follow the existing function style where each process returns a tensor of shape `(vocab_size, num_states, num_states)`.
2. Register the function in either `HMM_MATRIX_FUNCTIONS` or `GHMM_MATRIX_FUNCTIONS` depending on whether your process is a standard HMM or a generalised HMM.
3. Use the builders in `simplexity/generative_processes/builder.py` to construct the process. This ensures parameter validation and consistent initial state handling.
4. Add unit tests under `tests/generative_processes` to validate probability distributions and state transitions.

### Custom Processes

The library does not restrict generative processes to HMM-style transition matrices. To implement a completely new process, create a class deriving from `GenerativeProcess` (see `generative_processes/generative_process.py`). Implement the required methods:

```python
class MyProcess(GenerativeProcess[MyState]):
    @property
    def vocab_size(self) -> int: ...

    @property
    def initial_state(self) -> MyState: ...

    def emit_observation(self, state: MyState, key: chex.PRNGKey) -> chex.Array: ...

    def transition_states(self, state: MyState, obs: chex.Array) -> MyState: ...

    def observation_probability_distribution(self, state: MyState) -> jax.Array: ...

    def log_observation_probability_distribution(self, log_state: MyState) -> jax.Array: ...

    def probability(self, observations: jax.Array) -> jax.Array: ...

    def log_probability(self, observations: jax.Array) -> jax.Array: ...
```

This approach allows arbitrary internal state and emission logic. You can still use the training and evaluation utilities as long as your class conforms to the interface.

## Implementing a New Predictive Model

1. Define your model in `simplexity/predictive_models/`. Models should implement the `PredictiveModel` protocol found in `predictive_model.py`.
2. If the model has trainable parameters, make it an `equinox.Module` so that parameters can be updated functionally.
3. Provide a factory function (e.g. `build_my_model`) that constructs the model given a vocabulary size and random seed.
4. Add configuration options under `simplexity/configs/predictive_model` if the model requires specific hyperparameters.
5. Write unit tests in `tests/predictive_models`.

## Training Loops

Training utilities live under `simplexity/training`. If you create a new loop, mirror the call signature of `train_equinox_model.train()` so that configuration files and loggers remain interchangeable.

## Persistence and Logging

If a new persister or logger is required, implement it alongside the existing modules in `simplexity/persistence` or `simplexity/logging`. Both systems use simple abstract base classes, so minimal boilerplate is needed.

## Keeping Documentation in Sync

Whenever you add a significant new module, update this documentation folder to describe its purpose and usage. Clear docs make it easier for automated agents to stay aligned with the code.
