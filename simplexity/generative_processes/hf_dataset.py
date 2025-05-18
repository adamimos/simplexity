from __future__ import annotations

from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer

from .generative_process import GenerativeProcess


class HFDatasetProcess(GenerativeProcess[jax.Array]):
    """A generative process backed by a HuggingFace dataset."""

    dataset: jax.Array
    _vocab_size: int
    sequence_len: int
    _initial_state: jax.Array

    def __init__(self, data: Sequence[Sequence[int]], vocab_size: int):
        array = jnp.array(data, dtype=jnp.int32)
        if array.ndim != 2:
            raise ValueError("data must be a 2D array of token ids")
        self.dataset = array
        self._vocab_size = vocab_size
        self.sequence_len = array.shape[1]
        self._initial_state = jnp.zeros((1,), dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:  # noqa: D401 - short description
        """Return the size of the vocabulary."""
        return self._vocab_size

    @property
    def initial_state(self) -> jax.Array:  # noqa: D401 - short description
        """Return the initial state of the process (unused)."""
        return self._initial_state

    def emit_observation(self, state: jax.Array, key: chex.PRNGKey) -> jax.Array:
        """Unsupported for :class:`HFDatasetProcess`."""
        raise NotImplementedError("HFDatasetProcess does not support stepwise generation")

    def transition_states(self, state: jax.Array, obs: chex.Array) -> jax.Array:
        """Return the state unchanged."""
        return state

    def observation_probability_distribution(self, state: jax.Array) -> jax.Array:
        """Not implemented."""
        raise NotImplementedError("HFDatasetProcess does not implement probabilities")

    def log_observation_probability_distribution(self, log_belief_state: jax.Array) -> jax.Array:
        """Not implemented."""
        raise NotImplementedError("HFDatasetProcess does not implement probabilities")

    def probability(self, observations: jax.Array) -> jax.Array:
        """Not implemented."""
        raise NotImplementedError("HFDatasetProcess does not implement probabilities")

    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Not implemented."""
        raise NotImplementedError("HFDatasetProcess does not implement probabilities")

    # ------------------------------------------------------------------
    # Batched generation
    # ------------------------------------------------------------------
    def generate(
        self,
        state: jax.Array,
        key: chex.PRNGKey,
        sequence_len: int,
        return_all_states: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """Return a batch of sequences from the dataset."""
        if sequence_len > self.sequence_len:
            raise ValueError("Requested sequence length larger than dataset sequences")
        batch_size = state.shape[0]
        indices = jax.random.randint(key, (batch_size,), 0, self.dataset.shape[0])
        sequences = self.dataset[indices, :sequence_len]
        if return_all_states:
            states = jnp.repeat(state[:, None, :], sequence_len, axis=1)
            return states, sequences
        return state, sequences


def build_hf_dataset_process(
    dataset_name: str,
    *,
    split: str = "train",
    text_field: str = "text",
    tokenizer_name: str,
    sequence_len: int,
) -> HFDatasetProcess:
    """Build a :class:`HFDatasetProcess` from a HuggingFace dataset."""
    raw_ds = load_dataset(dataset_name, split=split)
    texts = raw_ds[text_field]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized = tokenizer(texts, add_special_tokens=False)["input_ids"]

    filtered = [seq[:sequence_len] for seq in tokenized if len(seq) >= sequence_len]
    vocab_size = int(tokenizer.vocab_size)
    return HFDatasetProcess(filtered, vocab_size)
