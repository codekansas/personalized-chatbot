"""Defines a task for fine-tuning an LLM on Facebook Messenger conversations."""

import logging
from dataclasses import dataclass

import ml.api as ml
from ml.core.state import Phase
from ml.tasks.base import DataLoaderConfig
from pretrained.rwkv import get_tokenizer
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from chatbot.models.llm import ChatbotModel
from chatbot.tasks.datasets.chatbot import ChatbotDataset

EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|padding|>"

logger = logging.getLogger(__name__)


@dataclass
class ChatbotTaskConfig(ml.SupervisedLearningTaskConfig):
    tsz: int = ml.conf_field(512, help="The maximum number of tokens in a sequence.")


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Model = ChatbotModel
Batch = tuple[Tensor, Tensor]
Output = Tensor
Loss = Tensor


@ml.register_task("chatbot", ChatbotTaskConfig)
class ChatbotTask(ml.SupervisedLearningTask[ChatbotTaskConfig, Model, Batch, Output, Loss]):
    def __init__(self, config: ChatbotTaskConfig) -> None:
        super().__init__(config)

        self.tokenizer = get_tokenizer()

        # Gets the token IDs for the special tokens.
        assert isinstance(eos_token := self.tokenizer.token_to_id(EOS_TOKEN), int)
        assert isinstance(pad_token := self.tokenizer.token_to_id(PAD_TOKEN), int)

        def tokenize(text: str) -> list[int]:
            return self.tokenizer.encode(text).ids

        self._dataset = ChatbotDataset(
            tsz=config.tsz,
            tokenizer=tokenize,
            eos_token=eos_token,
            pad_token=pad_token,
            tokenizer_key="rwkv_tokens",
        )

    def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
        tokens, _ = batch
        return model(tokens)

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        breakpoint()

        asdf

    def get_dataset(self, phase: ml.Phase) -> ChatbotDataset:
        return self._dataset

    def get_sampler(self, dataset: Dataset, cfg: DataLoaderConfig, phase: Phase) -> Sampler[int]:
        return self._dataset.get_sampler(cfg.batch_size)


def test_task_adhoc() -> None:
    ml.configure_logging()

    config = ChatbotTaskConfig()
    task = ChatbotTask(config)
    dataset = task.get_dataset("train")

    for i in range(3):
        sample, is_me = dataset[i]
        sample_str = task.tokenizer.decode(sample.tolist())
        logger.info("Sample: %s", sample_str)
        logger.info("Sample size: %s", sample.shape)
        logger.info("Is me: %s", is_me.shape)


if __name__ == "__main__":
    # python -m chatbot.tasks.chatbot
    test_task_adhoc()
