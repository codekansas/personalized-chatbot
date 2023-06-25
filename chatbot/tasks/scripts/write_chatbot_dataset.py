"""Generates the chatbot dataset for a desired tokenizer.

This script is used to convert the messenger conversation data to a dataset
that can be used with the chatbot model.
"""

import argparse
import json
import logging
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, get_args

import ftfy
import ml.api as ml
import tqdm

from chatbot.tasks.datasets.chatbot import TokenizerKey, _packed_file_path, get_tokenizer

logger = logging.getLogger(__name__)


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


def write_dataset(
    tokenizer: Callable[[str], list[int]],
    vocab_size: int,
    file_key: TokenizerKey,
    self_prefix: str | None,
    other_prefix: str | None,
    sep_str: str,
    empty_str: str,
    min_tokens_per_convo: int,
    max_tokens_per_convo: int,
    seconds_between_convos: int,
    compressed: bool,
    overwrite: bool,
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
        vocab_size: The number of possible tokens from the tokenizer.
        file_key: The key to use for the packed file.
        self_prefix: The prefix to use for messages sent by the user. If None,
            just use the original person's name.
        other_prefix: The prefix to use for messages sent by the other user.
            If None, just use the original person's name.
        sep_str: The token to use to separate messages.
        empty_str: The string to use for empty messages.
        min_tokens_per_convo: The minimum number of tokens in a conversation.
        max_tokens_per_convo: The maximum number of tokens in a conversation.
        seconds_between_convos: The minimum number of seconds between
            conversations.
        compressed: Whether to compress the packed file.
        overwrite: Whether to overwrite the packed file if it already exists.

    Returns:
        The path to the packed file.
    """
    if (packed_file_path := _packed_file_path(file_key)).exists():
        if overwrite:
            logger.warning("Overwriting existing packed file: %s", packed_file_path)
        else:
            raise FileExistsError(f"Packed file already exists: {packed_file_path} (use --overwrite to overwrite)")

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

    def get_name(name: str) -> str:
        if name == user_name:
            return f"{name}: " if self_prefix is None else self_prefix
        return f"{name}: " if other_prefix is None else other_prefix

    def get_message(m: dict[str, Any]) -> str:
        return get_name(m["sender_name"]) + ftfy.fix_text(m.get("content", empty_str))

    # Function for separating message histories into conversations.
    def gen_convos(messages: list[dict[str, Any]]) -> Iterator[list[int]]:
        split_ms = seconds_between_convos * 1000
        last_ms: int | None = None

        convo: list[str] = []
        tokens: list[int] = []
        for m in messages:
            if "timestamp_ms" not in m:
                convo = []
                tokens = []
                continue

            # Separate conversations by time.
            cur_ms = m["timestamp_ms"]
            if last_ms is not None and cur_ms - last_ms > split_ms:
                if len(tokens) > min_tokens_per_convo:
                    yield tokens
                convo = []
                tokens = []
            last_ms = cur_ms

            # Separate conversations by length.
            m_str = get_message(m)
            convo.append(m_str)
            tokens_next = tokenizer(sep_str.join(convo))
            if len(tokens_next) > max_tokens_per_convo:
                yield tokens
                convo = [m_str]
            else:
                tokens = tokens_next

        # Yield the last conversation.
        if len(tokens) > min_tokens_per_convo:
            yield tokens

    # Packs the training samples into a single file.
    with ml.Timer(f"writing packed file to {packed_file_path}"), ml.TokenWriter(
        (tmp_packed_file_path := packed_file_path.parent / f"{file_key}.bin.tmp"),
        num_tokens=vocab_size,
        compressed=compressed,
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

            for tokens in gen_convos(ordered_messages):
                writer.write(tokens)

    tmp_packed_file_path.rename(packed_file_path)

    return packed_file_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the chatbot dataset.")
    parser.add_argument(
        "mode",
        choices=get_args(TokenizerKey),
        help="The key to use for the packed file.",
    )
    parser.add_argument(
        "--self-prefix",
        type=str,
        help="The prefix to use for messages sent by the user.",
    )
    parser.add_argument(
        "--other-prefix",
        type=str,
        help="The prefix to use for messages sent by the other user.",
    )
    parser.add_argument(
        "--sep-str",
        type=str,
        default="\n",
        help="The token to use to separate messages.",
    )
    parser.add_argument(
        "--empty-str",
        type=str,
        default="<empty>",
        help="The string to use for empty messages.",
    )
    parser.add_argument(
        "--min-tokens-per-convo",
        type=int,
        default=128,
        help="The minimum number of tokens per conversation.",
    )
    parser.add_argument(
        "--max-tokens-per-convo",
        type=int,
        default=512,
        help="The maximum number of tokens per conversation.",
    )
    parser.add_argument(
        "--seconds-between-convos",
        type=int,
        default=60 * 60 * 8,
        help="The number of seconds between conversations.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Whether to compress the packed file.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the packed file if it already exists.",
    )
    args = parser.parse_args()

    # Gets the mode from the tokenizer.
    tokenizer, _, vocab_size, _ = get_tokenizer(args.mode)

    write_dataset(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        file_key=args.mode,
        self_prefix=args.self_prefix,
        other_prefix=args.other_prefix,
        sep_str=args.sep_str,
        empty_str=args.empty_str,
        min_tokens_per_convo=args.min_tokens_per_convo,
        max_tokens_per_convo=args.max_tokens_per_convo,
        seconds_between_convos=args.seconds_between_convos,
        compressed=args.compressed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    # python -m chatbot.tasks.scripts.write_chatbot_dataset
    main()
