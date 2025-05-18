import jax
import jax.numpy as jnp
import pytest
from datasets import Dataset

from simplexity.generative_processes.hf_dataset import HFDatasetProcess, build_hf_dataset_process


class DummyTokenizer:
    def __init__(self) -> None:
        self.vocab = {"hello": 0, "world": 1, "foo": 2, "bar": 3}
        self.vocab_size = len(self.vocab)

    def __call__(self, texts, add_special_tokens=False):
        ids = []
        for text in texts:
            ids.append([self.vocab.get(w, 0) for w in text.split()])
        return {"input_ids": ids}


def get_dummy_dataset() -> Dataset:
    return Dataset.from_dict({"text": ["hello world", "foo bar", "hello foo bar world"]})


def build_process(monkeypatch: pytest.MonkeyPatch) -> HFDatasetProcess:
    monkeypatch.setattr(
        "simplexity.generative_processes.hf_dataset.load_dataset",
        lambda name, split="train": get_dummy_dataset(),
    )
    monkeypatch.setattr(
        "simplexity.generative_processes.hf_dataset.AutoTokenizer.from_pretrained",
        lambda name: DummyTokenizer(),
    )
    return build_hf_dataset_process(
        dataset_name="dummy",
        split="train",
        text_field="text",
        tokenizer_name="dummy",
        sequence_len=3,
    )


def test_build_hf_dataset_process(monkeypatch: pytest.MonkeyPatch):
    process = build_process(monkeypatch)
    assert isinstance(process, HFDatasetProcess)
    assert process.dataset.shape == (3, 3)
    assert process.vocab_size == 4


def test_generate(monkeypatch: pytest.MonkeyPatch):
    process = build_process(monkeypatch)
    batch_size = 5
    state = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    states, sequences = process.generate(state, key, sequence_len=3, return_all_states=True)
    assert sequences.shape == (batch_size, 3)
    assert states.shape == (batch_size, 3, 1)
