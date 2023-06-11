"""Defines a dataset for reading from your Facebook Messenger data."""

import functools
import json
import logging
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

import ftfy
import ml.api as ml
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

logger = logging.getLogger(__name__)


@functools.lru_cache
def _get_message_paths(search_dir: Path) -> list[Path]:
    """Extracts the paths to all message files from the Facebook Messenger data.

    Args:
        search_dir: The directory to search for the Facebook Messenger data.

    Returns:
        A list of paths to all message files.
    """
    # Downloaded data should be called `facebook-{username}.zip`
    candidate_paths = list(search_dir.glob("facebook-*.zip"))
    if len(candidate_paths) == 0:
        raise FileNotFoundError(f"Could not find any Facebook Messenger data in {search_dir}")
    if len(candidate_paths) > 1:
        raise FileNotFoundError(f"Found multiple Facebook Messenger data files in {search_dir}")
    dataset_path = candidate_paths[0]

    # Unzips the data if it hasn't been unzipped already.
    if not (messages_path := dataset_path.parent / "messages").exists():
        with ml.Timer("extracting zipfiles"), zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path.parent)

        if not messages_path.exists():
            raise FileNotFoundError(f"The archive does not contain message data: {dataset_path}")

    # Gets the paths to all message files in the unzipped directory.
    inbox_dir = dataset_path.parent / "messages" / "inbox"
    if (cache_file := dataset_path.parent / ".messages.json").exists():
        with ml.Timer("loading cached messages"), open(cache_file, "r") as f:
            messages = [inbox_dir / p for p in json.load(f)]
    else:
        with ml.Timer("getting cached messages"):
            messages = sorted(list(inbox_dir.glob("*/message*.json")))
            assert len(messages) > 0, "No message files found"
            with open(cache_file, "w") as f:
                json.dump([str(p.relative_to(inbox_dir)) for p in messages], f, indent=2)

    return messages


@functools.lru_cache
def _packed_file_path(file_key: str) -> Path:
    search_dir = (ml.get_data_dir() / "messenger").resolve()
    if not search_dir.exists():
        raise FileNotFoundError(f"Could not find messenger data directory: {search_dir}")
    packed_file_path = search_dir / "packed" / f"{file_key}.bin"
    return packed_file_path


@functools.lru_cache
def _pack_training_samples(
    tokenizer: Callable[[str], list[int]],
    num_tokens: int,
    file_key: str,
    self_prefix: str = "Me: ",
    other_prefix: str = "Them: ",
    sep_str: str = "\n",
    empty_str: str = "<empty>",
    seconds_between_convos: int = 60 * 60 * 8,
    min_convo_length: int = 3,
) -> Path:
    """Packs training samples into a single file.

    This function generates a single file with consistent file offsets,
    containing the training samples parsed from the messenger data.

    Each conversation begins with a 32-bit integer with the total byte size
    of the packed conversation data, followed by a 32-bit integer with the
    number of tokens in the conversation, followed by the tokens, followed by
    a mask of values where 0 means it is part of a prefix, 1 means it was
    sent by the user, and 2 means it was sent by someone else.

    Args:
        tokenizer: A function that converts a string into a list of tokens.
        num_tokens: The number of possible tokens from the tokenizer.
        file_key: The key to use for the packed file.
        self_prefix: The prefix to use for messages sent by the user.
        other_prefix: The prefix to use for messages sent by the other user.
        sep_str: The token to use to separate messages.
        empty_str: The string to use for empty messages.
        token_dtype: The dtype to use for the tokens.
        seconds_between_convos: The number of seconds between conversations.
        min_convo_length: The minimum number of messages in a conversation.

    Returns:
        The path to the packed file.
    """
    if (packed_file_path := _packed_file_path(file_key)).exists():
        return packed_file_path
    packed_file_path.parent.mkdir(exist_ok=True)

    if ml.get_worker_info().in_worker or ml.get_world_size() > 1:
        raise RuntimeError("You should pre-process the dataset before using it.")

    # Gets the paths to all message files.
    all_messages = _get_message_paths(packed_file_path.parent.parent)

    # First iterates through all the messages to try to parse the user's ID.
    participant_counts: Counter[str] = Counter()
    with ml.Timer("determining user ID"):
        for messages_path in all_messages:
            with open(messages_path, "r") as f:
                messages = json.load(f)
            if "participants" not in messages or len(participants := messages["participants"]) <= 1:
                continue
            participant_counts.update([p["name"] for p in participants if "name" in p])
            if participant_counts.total() > 100:
                break
    if len(participant_counts) == 0:
        raise RuntimeError("Could not determine user ID")
    user_name = participant_counts.most_common(1)[0][0]
    logger.info("User name: %s", user_name)

    # Function for separating message histories into conversations.
    def gen_convos(messages: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
        split_ms = seconds_between_convos * 1000
        convo: list[dict[str, Any]] = []
        for message in messages:
            if "timestamp_ms" not in message:
                continue
            if len(convo) > 0 and message["timestamp_ms"] - convo[-1]["timestamp_ms"] > split_ms:
                if len(convo) > min_convo_length:
                    yield convo
                convo = []
            convo.append(message)
        if len(convo) > min_convo_length:
            yield convo

    # Packs the training samples into a single file.
    with ml.Timer("writing packed file"), ml.TokenWriter(
        (tmp_packed_file_path := packed_file_path.parent / f"{file_key}.bin.tmp"),
        num_tokens=num_tokens,
        compressed=True,
        overwrite_if_exists=True,
    ) as writer:
        for messages_path in tqdm.tqdm(all_messages):
            with open(messages_path, "r") as f:
                messages = json.load(f)
            if (
                "participants" not in messages
                or len(messages["participants"]) <= 1
                or all(p["name"] != user_name for p in messages["participants"])
            ):
                continue

            ordered_messages = sorted(messages["messages"], key=lambda m: m["timestamp_ms"])

            for convo in gen_convos(ordered_messages):
                full_convo = sep_str.join(
                    (
                        (self_prefix if m["sender_name"] == user_name else other_prefix)
                        + ftfy.fix_text(m.get("content", empty_str))
                    )
                    for m in convo
                )
                writer.write(tokenizer(full_convo))

    tmp_packed_file_path.rename(packed_file_path)

    return packed_file_path


def _get_sampler(offsets: list[int], num_samples: int) -> WeightedRandomSampler:
    weights = (np.diff(offsets) - 8).clip(min=0.0).astype(np.double)
    weights = (weights / (weights.sum() + 1e-3)).tolist()
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


class ChatbotDataset(Dataset[Tensor]):
    """Defines a dataset for training the chatbot.

    This dataset returns a tuple representing a conversation. The first element
    is the tokenized conversation, and the second element is a mask indicating
    which tokens are from the user, where 0 indicates a prompt token, 1
    indicates the user's tokens, 2 indicates another person's tokens, and 3
    indicates padding.
    """

    def __init__(
        self,
        tsz: int,
        tokenizer: Callable[[str], list[int]],
        num_tokens: int,
        pad_token: int,
        self_prefix: str = "Me: ",
        other_prefix: str = "Them: ",
        sep_token: str = "\n",
        tokenizer_key: str = "default",
        tokenizer_is_int32: bool = False,
    ) -> None:
        super().__init__()

        self._tsz = tsz
        self._pad_token = pad_token
        self._tok_bytes = 4 if tokenizer_is_int32 else 2
        self._tok_dtype = "I" if tokenizer_is_int32 else "H"

        packed_file_path = _pack_training_samples(
            tokenizer=tokenizer,
            num_tokens=num_tokens,
            file_key=tokenizer_key,
            token_dtype=self._tok_dtype,
            self_prefix=self_prefix,
            other_prefix=other_prefix,
            sep_str=sep_token,
        )

        self._reader = ml.TokenReader(
            packed_file_path,
            offsets_path=packed_file_path.with_suffix(".offsets"),
        )

    @classmethod
    def get_sampler(cls, file_key: str) -> WeightedRandomSampler:
        packed_file_path = _packed_file_path(file_key)
        reader = ml.TokenReader(
            packed_file_path,
            offsets_path=packed_file_path.with_suffix(".offsets"),
        )
        return _get_sampler(reader._offsets, 1000000)

    def __getitem__(self, index: int) -> Tensor:
        if index < 0:
            index += len(self)

        with open(self._packed_file_path, "rb") as fp:
            fp.seek(self._offsets[index] + 4, 0)
            length = np.frombuffer(fp.read(4), dtype="I").item()

            # Gets a random start position in the conversation.
            start = random.randint(0, max(length - self._tsz, 0))
            end = min(start + self._tsz, length)

            # Reads the tokens.
            fp.seek(start * self._tok_bytes, 1)
            tokens = np.frombuffer(fp.read((end - start) * self._tok_bytes), dtype=self._tok_dtype)

            # Pads the tokens at the end.
            tokens = np.pad(tokens, (0, self._tsz - len(tokens)), constant_values=self._pad_token)

        tokens_arr = torch.from_numpy(tokens.astype(np.int32))

        assert tokens_arr.shape == (self._tsz,)

        return tokens_arr

    def __len__(self) -> int:
        return len(self._offsets) - 1


def test_dataset_adhoc() -> None:
    ml.configure_logging()

    def simple_tokenizer(s: str) -> list[int]:
        return [ord(c) for c in s]

    def simple_detokenizer(t: list[int]) -> str:
        t = t[: t.index(0)] if 0 in t else t
        return "".join("\n" if c == 2 else chr(c) for c in t)

    dataset = ChatbotDataset(512, simple_tokenizer, 256, 0)
    for _ in range(5):
        sample = random.choice(dataset)
        logger.info("Sample: %s", simple_detokenizer(sample.tolist()))
        logger.info("Sample size: %s", sample.shape)


if __name__ == "__main__":
    # python -m chatbot.tasks.datasets.chatbot
    test_dataset_adhoc()
