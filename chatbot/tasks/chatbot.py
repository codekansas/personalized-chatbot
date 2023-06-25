"""Defines a task for fine-tuning an LLM on Facebook Messenger conversations."""

import itertools
import logging
from dataclasses import dataclass
from typing import cast, get_args

import ml.api as ml
import torch.nn.functional as F
from ml.core.state import Phase
from ml.tasks.base import DataLoaderConfig
from omegaconf import MISSING
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from chatbot.models.rwkv import RwkvChatbotModel
from chatbot.tasks.datasets.chatbot import ChatbotDataset, TokenizerKey, get_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class ChatbotTaskConfig(ml.SupervisedLearningTaskConfig):
    tsz: int = ml.conf_field(512, help="The maximum number of tokens in a sequence.")
    key: str = ml.conf_field(MISSING, help="The tokenizer key to use.")


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Model = RwkvChatbotModel
Batch = Tensor
Output = Tensor
Loss = dict[str, Tensor]


@ml.register_task("chatbot", ChatbotTaskConfig)
class ChatbotTask(ml.SupervisedLearningTask[ChatbotTaskConfig, Model, Batch, Output, Loss]):
    def __init__(self, config: ChatbotTaskConfig) -> None:
        super().__init__(config)

        # Gets the tokenizr and detokenizer.
        assert config.key in get_args(TokenizerKey), f"Invalid tokenizer key: {config.key}"
        self.key = cast(TokenizerKey, config.key)
        self._tokenize, self._detokenize, _, self._pad_token = get_tokenizer(cast(TokenizerKey, self.key))

    def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
        return model(batch[:, :-1])

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        preds, targets = output.transpose(1, 2), batch[:, 1:].long()

        # Token prediction loss.
        xent_loss = F.cross_entropy(preds, targets, ignore_index=self._pad_token, reduction="none")

        # Logs some samples.
        if state.phase == "valid":

            def show_gt() -> str:
                return model.tokens_to_string(batch[0])

            def sample_pred() -> str:
                prompt = "Them: Hey, it's been a while.\nThem: How are you doing?\nMe:"
                return prompt + "".join(list(model.infer(prompt)))

            self.logger.log_string("sample", show_gt)
            self.logger.log_string("pred", sample_pred)

        return {
            "token": xent_loss,
        }

    def get_dataset(self, phase: ml.Phase) -> ChatbotDataset:
        return ChatbotDataset(self.key, self.config.tsz, self._pad_token)

    def get_sampler(self, dataset: Dataset, cfg: DataLoaderConfig, phase: Phase) -> Sampler[int]:
        return ChatbotDataset.get_sampler(self.key)


def test_task_adhoc() -> None:
    ml.configure_logging()

    config = ChatbotTaskConfig()
    config.train_dl.batch_size = 2
    config.train_dl.num_workers = 0
    task = ChatbotTask(config)
    dataset = task.get_dataset("train")
    dataloader = task.get_dataloader(dataset, "train")

    for sample in itertools.islice(dataloader, 3):
        sample_str = task._detokenize(sample[0].tolist())
        logger.info("Sample: %s", sample_str)
        logger.info("Sample size: %s", sample.shape)


if __name__ == "__main__":
    # python -m chatbot.tasks.chatbot
    test_task_adhoc()
