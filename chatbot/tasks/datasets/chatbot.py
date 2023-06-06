"""Defines a dataset for reading from your Facebook Messenger data."""

import json
import logging
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

import ml.api as ml
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

SEPARATOR = "<|sep|>"


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


def _pack_training_samples(
    tokenizer: Callable[[str], Tensor],
    file_key: str,
    token_dtype: str = "i2",
    seconds_between_convos: int = 60 * 60 * 8,
    min_convo_length: int = 3,
) -> Path:
    """Packs training samples into a single file.

    This function generates a single file with consistent file offsets,
    containing the training samples parsed from the messenger data.

    Each conversation begins with a 64-bit integer with the total byte size
    of the packed conversation data, followed by a 32-bit integer with the
    number of messages N in the conversation, followed by N + 1 32-bit
    integers with the byte offsets of each message in the conversation,
    followed by N boolean values indicating whether the message was sent by the
    user, followed by the packed conversation data, containing the 16-bit or
    32-bit integer tokens for each message in the conversation.

    Args:
        tokenizer: A function that converts a string into a tensor.
        file_key: The key to use for the packed file.
        token_dtype: The dtype to use for the tokens.
        seconds_between_convos: The number of seconds between conversations.
        min_convo_length: The minimum number of messages in a conversation.

    Returns:
        The path to the packed file.
    """
    search_dir = (ml.get_data_dir() / "messenger").resolve()
    if not search_dir.exists():
        raise FileNotFoundError(f"Could not find messenger data directory: {search_dir}")
    packed_file_path = search_dir / "packed" / f"{file_key}.bin"
    if packed_file_path.exists():
        return packed_file_path
    packed_file_path.parent.mkdir(exist_ok=True)

    if ml.get_worker_info().in_worker:
        raise RuntimeError("You should pre-process the dataset before using it.")

    # Gets the paths to all message files.
    all_messages = _get_message_paths(search_dir)

    # First iterates through all the messages to try to parse the user's ID.
    participant_counts: Counter[str] = Counter()
    with ml.Timer("determining user ID"):
        for messages_path in all_messages:
            with open(messages_path, "r") as f:
                messages = json.load(f)
            if "participants" not in messages or len(participants := messages["participants"]) <= 1:
                continue
            participant_counts.update([p["name"] for p in participants if "name" in p])
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
    tmp_packed_file_path = packed_file_path.parent / f"{file_key}.bin.tmp"
    with ml.Timer("writing packed file"), open(tmp_packed_file_path, "wb") as fb:
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
                messages_contents = [(m.get("content", "")) for m in convo]
                tokens = [ml.as_numpy_array(tokenizer(m)) for m in messages_contents]
                offsets = np.cumsum([0] + [len(t) + 1 for t in tokens])
                is_me = np.array([m["sender_name"] == user_name for m in convo])

                length_bytes = np.array([len(tokens)], dtype="i4").tobytes()
                offsets_bytes = offsets.astype("i4").tobytes()
                is_me_bytes = is_me.astype("b1").tobytes()
                token_bytes = np.concatenate(tokens).astype(token_dtype).tobytes()
                total_bytes = len(length_bytes) + len(offsets_bytes) + len(is_me_bytes) + len(token_bytes)

                # Writes the conversation information to the file.
                fb.write(np.array([total_bytes], dtype="i8").tobytes())
                fb.write(length_bytes)
                fb.write(offsets_bytes)
                fb.write(is_me_bytes)
                fb.write(token_bytes)

    tmp_packed_file_path.rename(packed_file_path)

    return packed_file_path


def _compute_offsets(packed_file_path: Path) -> list[int]:
    """Computes the offsets of each conversation in the packed file.

    Args:
        packed_file_path: The path to the packed file.

    Returns:
        The offsets of each conversation in the packed file.
    """
    with ml.Timer("computing offsets"), open(packed_file_path, "rb") as f:
        offsets = [0]
        while True:
            try:
                length = np.frombuffer(f.read(8), dtype="i8").item()
                offsets.append(offsets[-1] + length + 8)
                f.seek(length, 1)
            except ValueError:
                break
    return offsets[:-1]


class ChatbotDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Defines a dataset for training the chatbot.

    This dataset returns a tuple representing a conversation. The first element
    is the tokenized conversation, the second element is a mask indicating which
    tokens are from the user, and the third element is a mask which has value 1
    if it is the start of the user's message, 2 if it is the start of another
    user's message, and 0 otherwise.
    """

    def __init__(
        self,
        tsz: int,
        tokenizer: Callable[[str], Tensor],
        pad_token: int,
        tokenizer_key: str = "default",
        tokenizer_is_int32: bool = False,
    ) -> None:
        super().__init__()

        self._tsz = tsz
        self._pad_token = pad_token
        self._tok_bytes = 4 if tokenizer_is_int32 else 2
        self._packed_file_path = _pack_training_samples(
            tokenizer=tokenizer,
            file_key=tokenizer_key,
            token_dtype="i4" if tokenizer_is_int32 else "i2",
        )
        self._offsets = _compute_offsets(self._packed_file_path)
        self._fp = open(self._packed_file_path, "rb")

    def __del__(self) -> None:
        if hasattr(self, "_fp"):
            self._fp.close()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if index < 0:
            index += len(self) - 1

        self._fp.seek(self._offsets[index] + 8, 0)
        length = np.frombuffer(self._fp.read(4), dtype="i4").item()
        offsets = np.frombuffer(self._fp.read(4 * length), dtype="i4")
        is_me = np.frombuffer(self._fp.read(length), dtype="b1")

        # Reads off some tokens.
        start = random.randint(0, max(offsets[-1] - self._tsz, 0))
        end = min(start + self._tsz, offsets[-1] - 1)
        self._fp.seek(start * self._tok_bytes, 1)
        tokens = np.frombuffer(self._fp.read((end - start) * self._tok_bytes), dtype=f"i{self._tok_bytes}")

        # Pads the tokens at the end.
        tokens = np.concatenate([tokens, np.full(self._tsz - len(tokens), self._pad_token, dtype=tokens.dtype)])

        # Gets the offsets for the start and end tokens.
        start_offset = np.searchsorted(offsets, start, side="right") - 1
        end_offset = np.searchsorted(offsets, end, side="right") - 1

        # Gets a mask for the tokens which are responses.
        is_response = np.zeros(self._tsz, dtype="i1")
        for i in range(start_offset, end_offset + 1):
            j = offsets[i] - start
            if j < 0:
                continue
            is_response[j] = 1 if is_me[i] else 2

        # Gets a mask for the tokens which are me.
        is_me_mask = [
            j for i in range(start_offset, end_offset + 1) for j in [is_me[i]] * (offsets[i + 1] - offsets[i])
        ]
        is_me_mask = is_me_mask[start - offsets[start_offset] : -(offsets[end_offset + 1] - end)]

        return torch.tensor(tokens), torch.tensor(is_me_mask), torch.tensor(is_response)

    def __len__(self) -> int:
        return len(self._offsets)


def test_dataset_adhoc() -> None:
    ml.configure_logging()

    def simple_tokenizer(s: str) -> Tensor:
        return torch.tensor([ord(c) for c in s])

    def simple_detokenizer(t: Tensor) -> str:
        return "".join(chr(c) for c in t)

    dataset = ChatbotDataset(512, simple_tokenizer, 0)
    sample, is_me, is_response = dataset[len(dataset) - 1]
    logger.info("Sample: %s", simple_detokenizer(sample))
    logger.info("Is me: %s", is_me)
    logger.info("Is response: %s", is_response)


if __name__ == "__main__":
    # python -m chatbot.tasks.dataset
    test_dataset_adhoc()
