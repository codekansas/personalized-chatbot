"""Defines a task for fine-tuning an LLM on Facebook Messenger conversations."""

import itertools
import logging
from dataclasses import dataclass

import ml.api as ml
import torch.nn.functional as F
from ml.core.state import Phase
from ml.tasks.base import DataLoaderConfig
from pretrained.rwkv import get_tokenizer
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from chatbot.models.llm import ChatbotModel
from chatbot.tasks.datasets.chatbot import ChatbotDataset

PAD_TOKEN = "<|padding|>"

FILE_KEY = "rwkv_tokens"

logger = logging.getLogger(__name__)


@dataclass
class ChatbotTaskConfig(ml.SupervisedLearningTaskConfig):
    tsz: int = ml.conf_field(64, help="The maximum number of tokens in a sequence.")
    supervise_prompt: bool = ml.conf_field(True, help="If set, supervise the prompt tokens as well.")
    supervise_other: bool = ml.conf_field(True, help="If set, supervise the other speaker's tokens as well.")


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Model = ChatbotModel
Batch = Tensor
Output = Tensor
Loss = dict[str, Tensor]


@ml.register_task("chatbot", ChatbotTaskConfig)
class ChatbotTask(ml.SupervisedLearningTask[ChatbotTaskConfig, Model, Batch, Output, Loss]):
    def __init__(self, config: ChatbotTaskConfig) -> None:
        super().__init__(config)

        self.tokenizer = get_tokenizer()
        self.supervise_prompt = config.supervise_prompt
        self.supervise_other = config.supervise_other

        # Gets the token IDs for the special tokens.
        assert isinstance(pad_token := self.tokenizer.token_to_id(PAD_TOKEN), int)
        self._pad_token = pad_token

    def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
        return model(batch[:, :-1])

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        preds, targets = output.transpose(1, 2), batch[:, 1:].long()

        # model.tokens_to_string(preds.argmax(dim=1)[0])
        # model.tokens_to_string(targets[0])

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
        return ChatbotDataset(
            tsz=self.config.tsz,
            tokenizer=lambda text: self.tokenizer.encode(text).ids,
            num_tokens=self.tokenizer.get_vocab_size(),
            pad_token=self._pad_token,
            tokenizer_key=FILE_KEY,
        )

    def get_sampler(self, dataset: Dataset, cfg: DataLoaderConfig, phase: Phase) -> Sampler[int]:
        return ChatbotDataset.get_sampler(FILE_KEY)


def test_task_adhoc() -> None:
    ml.configure_logging()

    config = ChatbotTaskConfig()
    config.train_dl.batch_size = 2
    config.train_dl.num_workers = 0
    task = ChatbotTask(config)
    dataset = task.get_dataset("train")
    dataloader = task.get_dataloader(dataset, "train")

    for sample in itertools.islice(dataloader, 3):
        sample_str = task.tokenizer.decode(sample[0].tolist())
        logger.info("Sample: %s", sample_str)
        logger.info("Sample size: %s", sample.shape)


if __name__ == "__main__":
    # python -m chatbot.tasks.chatbot
    test_task_adhoc()
