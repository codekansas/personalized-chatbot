"""Defines a dataset for reading from your Facebook Messenger data."""

import functools
import logging
import random
from pathlib import Path
from typing import Callable, Literal

import ml.api as ml
import numpy as np
import torch
from pretrained.rwkv import get_tokenizer as get_rwkv_tokenizer
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

logger = logging.getLogger(__name__)

TokenizerKey = Literal["rwkv"]


@functools.lru_cache
def _packed_file_path(file_key: str) -> Path:
    search_dir = (ml.get_data_dir() / "messenger").resolve()
    if not search_dir.exists():
        raise FileNotFoundError(f"Could not find messenger data directory: {search_dir}")
    packed_file_path = search_dir / "packed" / f"{file_key}.bin"
    return packed_file_path


def _get_sampler(offsets: list[int], num_samples: int) -> WeightedRandomSampler:
    weights = (np.diff(offsets) - 8).clip(min=0.0).astype(np.double)
    weights = (weights / (weights.sum() + 1e-3)).tolist()
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


def get_tokenizer(key: TokenizerKey) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str], int, int]:
    """For a given tokenizer key, returns the tokenizer, detokenizer, vocab size and pad token.

    Args:
        key: The tokenizer key.

    Returns:
        A tuple containing the tokenizer, detokernizer, vocab size and pad
        token. The tokenizer function just takes a string as input and outputs
        a list of integers.
    """
    match key:
        case "rwkv":
            rwkv_tokenizer = get_rwkv_tokenizer()
            assert isinstance(vocab_size := rwkv_tokenizer.get_vocab_size(), int)
            assert isinstance(pad_token := rwkv_tokenizer.token_to_id("<|padding|>"), int)
            return lambda text: rwkv_tokenizer.encode(text).ids, rwkv_tokenizer.decode, vocab_size, pad_token
        case _:
            raise NotImplementedError(f"Mode {key} not implemented")


class ChatbotDataset(Dataset[Tensor]):
    """Defines a dataset for training the chatbot.

    This dataset returns a tuple representing a conversation. The first element
    is the tokenized conversation, and the second element is a mask indicating
    which tokens are from the user, where 0 indicates a prompt token, 1
    indicates the user's tokens, 2 indicates another person's tokens, and 3
    indicates padding.

    Parameters:
        key: The tokenizer key.
        tsz: The maximum number of tokens in a sample.
        pad_token: The padding token.
        in_memory: Whether to load the full dataset into memory.
        from_start: Whether to sample from the start of the conversation or
            from a random point in the conversation.
    """

    def __init__(
        self,
        key: str,
        tsz: int,
        pad_token: int,
        in_memory: bool = False,
        from_start: bool = True,
    ) -> None:
        super().__init__()

        self._tsz = tsz
        self._pad_token = pad_token
        self._from_start = from_start

        packed_file_path = _packed_file_path(key)
        self._reader = ml.TokenReader(packed_file_path, None, in_memory=in_memory)

    @classmethod
    def get_sampler(cls, file_key: str) -> WeightedRandomSampler:
        packed_file_path = _packed_file_path(file_key)
        reader = ml.TokenReader(packed_file_path, None)
        return _get_sampler(reader._offsets, 1000000)

    def __getitem__(self, index: int) -> Tensor:
        length = self._reader.length(index)
        start = 0 if self._from_start else random.randint(0, max(length - self._tsz, 0))
        end = min(start + self._tsz, length)
        tokens = self._reader[index, start:end] + [self._pad_token] * (self._tsz - (end - start))
        return torch.tensor(tokens)

    def __len__(self) -> int:
        return len(self._reader)


def test_dataset_adhoc(key: TokenizerKey = "rwkv") -> None:
    ml.configure_logging()

    dataset = ChatbotDataset(key, 512, 0)
    _, detokenizer, _, _ = get_tokenizer(key)
    for _ in range(5):
        sample = random.choice(dataset)
        logger.info("Sample: %s", detokenizer(sample.tolist()))
        logger.info("Sample size: %s", sample.shape)


if __name__ == "__main__":
    # python -m chatbot.tasks.datasets.chatbot
    test_dataset_adhoc()
